# Jasper based on Jeff
# 2021-11-20
# 基于2015年的Image Caption模型改写，使用Flickr8k数据集训练，添加双语注释
# liujiarun01@126.com

import os
import string
import glob
# import cv2

import tensorflow as tf
from tensorflow.keras.applications import MobileNet
import tensorflow.keras.applications.mobilenet 
from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow.keras.applications.inception_v3

from tqdm import tqdm # using it for generating visualized running process
import tensorflow.keras.preprocessing.image
import pickle
from time import time
import numpy as np
from PIL import Image, ImageFile

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector
from tensorflow.keras.layers import Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import Input, layers
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# ### 超参数

START = "startseq" # 开始标记
STOP = "endseq" # 终止标记
root_captioning = "." # 根目录
TRAIN = True # 训练或推理
dataset = 'Flickr' # 训练集名称 # Flickr, COCO
hdf5_path = './checkpoints/caption_model.hdf5' # 断点模型加载

# TODO: 超参数调参
EPOCHS = 2 # 迭代次数
dp = 0.5 # dropout概率
dense = 256 # 隐藏层大小
lr = 0.001 # adam 学习率
batch = 32 # 一批图片数量

USE_INCEPTION = True # 是否使用Inception卷积网络
cnn_net = 'imagenet' # encoder使用的网络
WIDTH = 299 # 输入图片的长
HEIGHT = 299 # 输入图片的宽
OUTPUT_DIM = 2048 # encoder层的特征向量的输出维度
USE_PKL = True # encode过程是否使用pkl文件

word_count_threshold = 2 # 低频词汇限制阈值
embedding_dim = 300 # 应该小于等于Glove的向量维度（300d.txt是300）


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"

# ---------------------------------------------------------
# ### 数据清洗
#
# ### 需要的数据
# 
# You will need to download the following data and place it in a folder for this example.  Point the *root_captioning* string at the folder that you are using for the caption generation. This folder should have the following sub-folders.
# 
# * data - Create this directory to hold saved models.
# * [glove.6B](https://nlp.stanford.edu/projects/glove/) - Glove embeddings.
# * [Flicker8k_Dataset](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip) - Flicker dataset.
# * [Flicker8k_Text](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip)
# ### 将注释的标点，空格等干扰项去除并整理

null_punct = str.maketrans('', '', string.punctuation) # 将字符串中的标点去除，自定义函数
lookup = dict() # 查找表，字典格式

with open(os.path.join(root_captioning,'data','Flickr8k_text','Flickr8k.token.txt'), 'r') as fp:
  
  max_length = 0
  for line in fp.read().split('\n'): # 对每一行遍历

    # TODO： 此处应添加训练集和测试集对应txt文件的构建，内容是图片名 xxx.jpg（按照比例随机抽取训练和测试集）

    tok = line.split()
    if len(line) >= 2:
      id = tok[0].split('.')[0] # 取出图片的id 例如'3630332976_fdba22c50b'
      desc = tok[1:] # 提取出图片的描述，一张图片对应五个不同的描述，所以相同的id可能对应不同的desc
      
      # 清洗描述
      desc = [word.lower() for word in desc] # 小写
      desc = [w.translate(null_punct) for w in desc] # 删除标点
      desc = [word for word in desc if len(word)>1] # 删除空格
      desc = [word for word in desc if word.isalpha()] # 删除阿拉伯数字和符号
      max_length = max(max_length,len(desc)) # 更新最长描述的长度
      
      if id not in lookup:
        lookup[id] = list() # 将id添加到查找表中
      lookup[id].append(' '.join(desc)) # 将描述作为value，图片id作为key，完善查找表
      
lex = set()
for key in lookup:
  [lex.update(d.split()) for d in lookup[key]] # lex集合包括了所有训练数据中的描述所用到的词

print(len(lookup)) # 图片总数，也即id数 8092张（flickr8k）
print(len(lex)) # 字典 8763个词（flickr8k）
print(max_length) # 最长的描述包含的词数 32个（flickr8k）

# 加载所有训练图片（flickr8k）
img = glob.glob(os.path.join(root_captioning,'data','Flicker8k_Dataset', '*.jpg'))
print(len(img)) # 8092张

# 从索引txt文件里读取训练集和测试集图片的文件名 #TODO: 这部分需要自动生成
train_images_path = os.path.join(root_captioning,'data','Flickr8k_text','Flickr_8k.trainImages.txt') 
train_images = set(open(train_images_path, 'r').read().strip().split('\n')) # 6000个id
test_images_path = os.path.join(root_captioning,'data','Flickr8k_text','Flickr_8k.testImages.txt') 
test_images = set(open(test_images_path, 'r').read().strip().split('\n')) # 1000个id

train_img = [] # 训练图片的可遍历对象
test_img = [] # 测试图片的可遍历对象

for i in img:
  f = os.path.split(i)[-1] # 获取图片名
  if f in train_images: # 如果名字在训练集
    train_img.append(f)
  elif f in test_images: # 如果名字在测试集
    test_img.append(f)

print(len(train_images))
print(len(test_images))

# 为训练集的每条描述加上开始和结束标记
# 之后会用start标记开始生成一段描述，当遇到stop标记则判断生成语句结束

train_descriptions = {k:v for k, v in lookup.items() if f'{k}.jpg' in train_images}
for n, v in train_descriptions.items(): 
  for d in range(len(v)): # len(v)通常为5，即5种不同的描述
    v[d] = f'{START} {v[d]} {STOP}' # 为每一句描述添加开始和结束

# the train_descriptions contains things like: 
# '963730324_0638534227': ['startseq kid is pushing shopping cart behind an adult inside consumer market endseq', 'startseq little boy is hanging on to shopping cart following behind man endseq', 'startseq small boy wearing white shirt pushes grey shopping cart endseq', 'startseq young boy attempts to push the walmart shopping cart for man endseq', 'startseq the young boy is pushing the cart inside the store endseq']
print(len(train_descriptions)) # 6000个训练数据

test_descriptions = {k:v for k, v in lookup.items() if f'{k}.jpg' in test_images}
for n, v in test_descriptions.items(): 
  for d in range(len(v)): # len(v)通常为5，即5种不同的描述
    v[d] = f'{START} {v[d]} {STOP}' # 为每一句描述添加开始和结束

# ---------------------------------------------------------
# ### 选择要迁移的CNN模型，即Encoder，并调整超参数
# 
# There are two neural networks that are accessed via transfer learning.  In this example, I use Glove for the text embedding and InceptionV3 to extract features from the images.  Both of these transfers serve to extract features from the raw text and the images. 
# 
# By setting the values WIDTH, HEIGHT, and OUTPUT_DIM you can interchange images.  One characteristic that you are seeking for the image neural network is that it does not have too many outputs (once you strip the 1000-class imagenet classifier, as is common in transfer learning).  
# InceptionV3 has 2,048 features below the classifier and MobileNet has over 50K.  If the additional dimensions truely capture aspects of the images, then they are worthwhile.  However, having 50K features increases the processing needed and the complexity of the neural network we are constructing.

# TODO: 选择不同的encoder，有DenseNet，resnet，详见tf官网
encode_model = InceptionV3(weights='imagenet') # cnn_net
encode_model = Model(encode_model.input, encode_model.layers[-2].output) # 取出倒数第二层的输出作为特征提取层
preprocess_input = tensorflow.keras.applications.inception_v3.preprocess_input # keras预处理输入的函数
encode_model.summary()


# ---------------------------------------------------------
# ### 基于特定的encoder和预处理层
# ### 将训练集和测试集的图片进行编码（encode）
# 

def encodeImage(img): # 训练过程，将图片预处理喂入encoder层，输出特征向量，维度为OUTPUT_DIM
  # Resize all images to a standard size (specified by the image encoding network)
  img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS) # resize image with high-quality and standard size
  # Convert a PIL image to a numpy array
  x = tensorflow.keras.preprocessing.image.img_to_array(img) # 1-D array
  # Expand to 2D array
  x = np.expand_dims(x, axis=0) # with a [ ] outside
  # Perform any preprocessing needed by InceptionV3 or others
  x = preprocess_input(x)
  # Call InceptionV3 (or other) to extract the smaller feature set for the image.
  x = encode_model.predict(x) # Get the encoding vector for the image
  # Shape to correct form to be accepted by LSTM captioning network.
  x = np.reshape(x, OUTPUT_DIM)
  return x

def encodeImageArray(img): # 只用于推理过程，同上函数功能相同
  img = tensorflow.keras.preprocessing.image.array_to_img(img) # 输入为cv2.capture()捕捉到的序列，需转换为img
  # Resize all images to a standard size (specified by the image encoding network)
  img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
  # Convert a PIL image to a numpy array
  x = tensorflow.keras.preprocessing.image.img_to_array(img)
  # Expand to 2D array
  x = np.expand_dims(x, axis=0)
  # Perform any preprocessing needed by InceptionV3 or others
  x = preprocess_input(x)
  # Call InceptionV3 (or other) to extract the smaller feature set for the image.
  x = encode_model.predict(x) # Get the encoding vector for the image
  # Shape to correct form to be accepted by LSTM captioning network.
  x = np.reshape(x, OUTPUT_DIM)
  return x

# We can now generate the training set.  This will involve looping over every JPG that was provided.  
# Because this can take awhile to perform we will save it to a pickle file.  This saves the considerable time needed to completly reprocess all of the images.  
# Because the images are processed differently by different transferred neural networks, the output dimensions are also made part of the file name(2048 for imagenet).  
# If you changed from InceptionV3 to MobileNet, the number of output dimensions would change, and a new file would be created.
# If you changed to other dataset, make sure to encode from your own data.

train_path = os.path.join(root_captioning, 'checkpoints', f'train{OUTPUT_DIM}_{dataset}.pkl') # 用OUTPUT_DIM和dataset命名

if USE_PKL: # 如果使用pkl文件
  if not os.path.exists(train_path):
    print('pkl file not exist.')
  else:
    with open(train_path, "rb") as fp:
      encoding_train = pickle.load(fp)

else: # 对于新的数据集，需要重新生成pkl文件，不使用pkl文件
  start = time()
  encoding_train = {} # 将编码结果存储到字典 key = id, value = 编码后的图片特征向量
  for id in tqdm(train_img): # 6000 imgs, 6000 id
    image_path = os.path.join(root_captioning, 'data', 'Flicker8k_Dataset', id)
    img = tensorflow.keras.preprocessing.image.load_img(image_path, target_size=(HEIGHT, WIDTH))
    encoding_train[id] = encodeImage(img) # predefined below to generate encoded vecs
  
  with open(train_path, "wb") as fp:
    pickle.dump(encoding_train, fp) # dump the whole train dict to pickle file
  print(f"\nGenerating training set took: {hms_string(time()-start)}") # time needed


# A similar process must also be performed for the test images.

if USE_PKL: 
  test_path = os.path.join(root_captioning, 'checkpoints', f'test{OUTPUT_DIM}_{dataset}.pkl') # 用OUTPUT_DIM命名
  if not os.path.exists(test_path): # 1000 iterations for encoding the test imgs
    print('pkl file not exist.')
  else:
    with open(test_path, "rb") as fp:
     encoding_test = pickle.load(fp)

else:
  start = time()
  encoding_test = {}
  for id in tqdm(test_img):
    image_path = os.path.join(root_captioning, 'data', 'Flicker8k_Dataset', id)
    img = tensorflow.keras.preprocessing.image.load_img(image_path, target_size=(HEIGHT, WIDTH))
    encoding_test[id] = encodeImage(img)

  with open(test_path, "wb") as fp:
    pickle.dump(encoding_test, fp)
  print(f"\nGenerating testing set took: {hms_string(time()-start)}")

# ---------------------------------------------------------
# ### 处理标注，构建词表
# ### 为构建embedding和decoder层做准备
# 
# so far we have the encoding_train and encoding_test dicts, containing 6k and 1k imgs vec outputs via inceptionV3 network
# Next we separate the captions that will be used for training. 
# There are two sides to this training, the images and the captions.

# 去除比较少见的词汇
# TODO:可以考虑舍去这一环节

all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val: # 1 img has 5 captions
        all_train_captions.append(cap)

word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
print('preprocessed words %d ==> %d' % (len(word_counts), len(vocab)))

# 构建注释中所有词汇的查找表，也相当于自然数编码
idxtoword = {} # 索引是id
wordtoidx = {} # 索引是词汇
ix = 1
for w in vocab:
    wordtoidx[w] = ix # {'startseq': 1, 'child': 2, 'in': 3, 'pink': 4, 'dress': 5, ...
    idxtoword[ix] = w # {1: 'startseq', 2: 'child', 3: 'in', 4: 'pink', 5: 'dress', ...
    ix += 1

# 词向量空间的大小，为后文embedding网络大小
vocab_size = len(idxtoword) + 1 
print(vocab_size)

# 考虑到有开始和结束标记，将最大长度增长2个
max_length += 2
print(max_length)


# ---------------------------------------------------------
# ### 构建数据生成器，并生成最终的训练数据和测试数据
# ### 将一句话中的一个词的前面所有词和对应的图片作为输入，将该词作为ground truth训练
# 
# Up to this point we've always generated training data ahead of time and fit the neural network to it.  It is not always practical to generate all of the training data ahead of time.  The memory demands can be considerable.  If the training data can be generated, as the neural network needs it, it is possable to use a Keras generator.  The generator will create new data, as it is needed.  The generator provided here creates the training data for the caption neural network, as it is needed.
# If we were to build all needed training data ahead of time it would look something like below.
# Here we are just training on two captions.  However, we would have to duplicate the image for each of these partial captions that we have.  Additionally the Flikr8K data set has 5 captions for each picture.  Those would all require duplication of data as well.  It is much more efficient to just generate the data as needed.

def data_generator(descriptions, photos, wordtoidx, max_length, num_photos_per_batch):
  # x1 - Training data for photos
  # x2 - The caption that goes with each photo
  # y - The predicted rest of the caption
  x1, x2, y = [], [], []
  n = 0
  while True:
    for key, desc_list in descriptions.items():
      n += 1
      photo = photos[key + '.jpg']
      for desc in desc_list:
        # Convert each word into a list of sequences.
        seq = [wordtoidx[word] for word in desc.split(' ') if word in wordtoidx]
        # Generate a training case for every possible sequence and outcome
        for i in range(1, len(seq)): # seq is like [3, 165, 40, 6, 28, 241, ...]
          in_seq, out_seq = seq[:i], seq[i] # using words sequence in_seq to predict 1 word out_seq, we generate caption training data here
          in_seq = pad_sequences([in_seq], maxlen=max_length)[0] # padding 0
          out_seq = to_categorical([out_seq], num_classes=vocab_size)[0] # one-hot seq, dimension of vocab_size
          
          x1.append(photo)
          x2.append(in_seq)
          y.append(out_seq)
      if n == num_photos_per_batch:
        yield ([np.array(x1), np.array(x2)], np.array(y)) 
        x1, x2, y = [], [], [] # 清空栈
        n = 0 # TODO:可以改成整除 不必清除n


# ---------------------------------------------------------
# ### 构建Glove的embedding层和词向量空间，作为decoder层

glove_dir = os.path.join(root_captioning, 'data', 'glove.6B')
embeddings_index = {} # 构建词向量空间
f = open(os.path.join(glove_dir, 'glove.6B.300d.txt'), encoding="utf-8")

for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs # here we have a dict, key = word, value = glove_tensor

f.close() 
print(f'Found {len(embeddings_index)} word vectors.') # 400001 here

# Get 200-dim dense vector for each of the 10000 words in out vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim)) # 基于数据集的embedding矩阵

for word, i in wordtoidx.items():
    #if i < max_words:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in the embedding index will be all zeros cause np.zeros() below
        embedding_matrix[i] = embedding_vector

# It is 1652 (the size of the vocabulary) by 200 (the number of features Glove generates for each word).
print(embedding_matrix.shape)


# ---------------------------------------------------------
# ### 搭建神经网络模型
# 
# An embedding matrix is built from Glove.  This will be directly copied to the weight matrix of the neural network.
class ICM(Model): # Image Caption Model的缩写，实际上只是decoder部分（encoder不需要训练）
  # TODO: 可以用Sequential和TimeDistributed模型改写
  def __init__(self):
    super(ICM, self).__init__()
    self.inputs1 = Input(shape=(OUTPUT_DIM,)) # 这里的输入已经是图片的特征向量
    self.inputs2 = Input(shape=(max_length,)) # 这里的输入是（某个特定词前的）词序列
    self.dropout = Dropout(dp) # 加入dropout防止过拟合
    self.dense = Dense(dense, activation='relu') # 稠密层
    self.dense_softmax = Dense(vocab_size, activation='softmax') # 输出分布层
    self.Embedding = Embedding(vocab_size, embedding_dim, mask_zero=True) # 嵌入层
    self.lstm = LSTM(dense) # lstm层，这里为单层 # TODO:双层lstm的论文复现
    # TODO:是否需要加入批标准化
    self.add = add

  def call(self, x1, x2):
    inputs1 = self.inputs1(x1)
    inputs2 = self.inputs2(x2)
    fe1 = self.dropout(inputs1)
    fe2 = self.dense(fe1)
    se1 = self.Embedding(inputs2)
    se2 = self.dropout(se1)
    se3 = self.lstm(dense)(se2) 
    decoder1 = self.add([fe2, se3]) # 将图片特征和语义序列特征拼接
    decoder2 = self.dense(decoder1) # 稠密层
    outputs = self.dense_softmax(decoder2) # softmax预测层，给出最大似然输出
    return outputs


# ---------------------------------------------------------
# ### 训练过程
# ### 保存模型断点到data
if Train:
  print('train and save model.')
  model = ICM()
  # 将模型的Embedding层的权重设置为提前做好的 embedding_matrix
  model.layers[2].set_weights([embedding_matrix]) 
  model.layers[2].trainable = False # 不训练该层（Glove的迁移层）
  # model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy']) # 优化器，可调参
  loss_object = tf.keras.losses.CategoricalCrossentropy() #from_logits=True
  optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

  model.summary()

  train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy') # acc指标
  test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

  @tf.function
  def train_step(x1, x2, y):
    with tf.GradientTape() as tape:
      predictions = model(x1, x2, training=True)
      loss = loss_object(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_accuracy(y, predictions)

  @tf.function
  def test_step(x1, x2, y):
    predictions = model(x1, x2, training=False)
    test_loss = loss_object(y, predictions)

    test_loss(test_loss)
    test_accuracy(labels, predictions)

  for epoch in range(EPOCHS): 
    train_accuracy.reset_states()
    test_accuracy.reset_states()

    # 通过yield实现iterable的batch批量训练
    for x1, x2, y in data_generator(train_descriptions, train_img, wordtoidx, max_length, batch):
      train_step(x1, x2, y)
      print(
        f'Epoch {epoch + 1}, '
        f'Train Loss: {train_loss.result()}, '
        f'Train Accuracy: {train_accuracy.result() * 100}, '       
      )

    # 每一轮验证，注意这里的测试集（test）实际上是指验证集（val）
    for x1, x2, y in data_generator(test_descriptions, test_img, wordtoidx, max_length, batch):
      test_step(x1, x2, y)
      print(
        f'Epoch {epoch + 1}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}'
      )

  saved_model_path = "./checkpoints/caption_model_{}.hdf5".format(int(time.time()))
  model.save(saved_model_path)


# ---------------------------------------------------------
# ### 推理过程
# 
# It is important to understand that a caption is not generated with one single call to the neural network's predict function.  Neural networks output a fixed-length tensor.  To get a variable length output, such as free-form text, requires multiple calls to the neural network.
# The neural network accepts two objects (which are mapped to the input neurons).  The first is the photo.  The second is an ever growing caption.  The caption begins with just the starting token.  The neural network's output is the prediction of the next word in the caption.  This continues until an end token is predicted or we reach the maximum length of a caption.  Each time predict a new word is predicted for the caption.  The word that has the highest probability (from the neural network) is chosen. 

def generateCaption(photo, model): # generate the prediction based on the model and given photo
  in_text = START
  for i in range(max_length):
      sequence = [wordtoidx[w] for w in in_text.split() if w in wordtoidx]
      sequence = pad_sequences([sequence], maxlen=max_length)
      yhat = model.predict([photo, sequence], verbose=0)
      yhat = np.argmax(yhat)
      word = idxtoword[yhat]
      in_text += ' ' + word
      if word == STOP:
          break
  final = in_text.split()
  final = final[1:-1] # delete the start / end token
  final = ' '.join(final)
  return final

def __draw_label(img, text, pos, bg_color):
  font_face = cv2.FONT_HERSHEY_TRIPLEX
  scale = 0.5
  color = (255, 255, 255)
  thickness = cv2.FILLED
  margin = 5

  txt_size = cv2.getTextSize(text, font_face, scale, thickness)

  end_x = pos[0] + txt_size[0][0] + margin
  end_y = pos[1] - txt_size[0][1] - margin

  cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
  cv2.putText(img, text, pos, font_face, scale, color, 2, cv2.LINE_AA)

if not TRAIN: # TODO: 可更改测试图片
  model_path = os.path.join(root_captioning,"checkpoints", hdf5_path)
  model = ICM()
  model.load_weights(model_path) # load the weights only, model already defined
  im_path = os.path.join(root_captioning, 'data', 'Flickr8k_Dataset', '3708244207_0d3a2b2f92.jpg')
  img = cv2.imread(im_path)

  plt.imshow(img)
  plt.show()

  print('---------')
  print('Caption:', generateCaption(img, model))

if not TRAIN:
  print('start inference.')

