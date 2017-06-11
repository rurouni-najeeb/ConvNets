## Visualising the fully connected codes using t-SNE
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time
import os
from tqdm import tqdm
import make_cnn as cnn
import csv
from tensorflow.contrib.tensorboard.plugins import projector
import matplotlib.pyplot as plt

## Defining the constants
IMAGE_DEPTH = 1
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
BATCH_SIZE = 100
N_EPOCHS = 5
LEARNING_RATE = 1e-4

checkpoint_dir='checkpoints/cnn'
embeddings_dir='checkpoints/embeddings'
metadata_filename=embeddings_dir+'/metadata.tsv'

## Meta data file should be under the embedding directory where the tf.summary.FileWriter is defined
def makeMetaData(labels, metadata_filename=metadata_filename):
    
    ## Writing characteristic of data to a tsv file
    with open(metadata_filename, 'wb') as fp:
        spamwriter = csv.writer(fp, delimiter='\t',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['Data','Label'])
        for i in range(len(labels)):
            spamwriter.writerow([i+1, labels[i]])

## Take from http://www.pinchofintelligence.com/simple-introduction-to-tensorboard-embedding-visualisation/
def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    
    
    spriteimage = np.ones((img_h * n_plots ,img_w * n_plots ))
    
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                  j * img_w:(j + 1) * img_w] = this_img
    
    return spriteimage

def vector_to_matrix_mnist(mnist_digits):
    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
    return np.reshape(mnist_digits,(-1,28,28))

def invert_grayscale(mnist_digits):
    """ Makes black white, and white black """
    return 1-mnist_digits


def visualiseEmbeddings(x_batch, y_batch, placeholders, layer, metadata_filename=metadata_filename, checkpoint_dir=checkpoint_dir,embeddings_dir=embeddings_dir):

    with tf.Session() as sess:
        
        image, label = placeholders
        ## Defining saver
        saver = tf.train.Saver()

        ## Restoring the checkpoint
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join(checkpoint_dir,'checkpoint')))
        if ckpt and ckpt.model_checkpoint_path:
            print 'Getting the checkpoint'
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        embed_matrix = sess.run(layer, feed_dict={image:x_batch.reshape([-1, 28, 28, 1]), label:y_batch})
        embedding_var = tf.Variable(embed_matrix, name="embeddings")
        sess.run(embedding_var.initializer)

        ## Embeddings
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = metadata_filename   

        ## Sprite Information
        embedding.sprite.image_path = embeddings_dir+'/sprite_image.png'
        embedding.sprite.single_image_dim.extend([28,28])

        ## Visualising the embeddings
        writer = tf.summary.FileWriter(embeddings_dir)
        projector.visualize_embeddings(writer, config)

        ## Saving the embeddings
        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(sess, embeddings_dir+'/embeddings.ckpt', 1)

def main():
    
    ## Initialising the input layer
    with tf.variable_scope("input") as scope:
        image = tf.placeholder(dtype=tf.float32,shape=[None,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH],name="image")
        y = tf.placeholder(dtype=tf.float32,shape=[None,10],name="label")
    
    ## Reading in the data and defining constants
    print 'Reading in the data...'
    mnist = input_data.read_data_sets("../MNIST_data/",one_hot=True)
    n_training_size,n_features = mnist.train.images.shape
    x_batch = mnist.train.images[0:500,:]
    y_batch = mnist.train.labels[0:500]

    ## Making the metadata file
    print 'Making metadata file'
    makeMetaData(np.argmax(y_batch,1))

    ## Making the sprite image
    to_visualise = vector_to_matrix_mnist(x_batch)
    to_visualise = invert_grayscale(to_visualise)

    sprite_image = create_sprite_image(to_visualise)

    plt.imsave(embeddings_dir+'/sprite_image.png',sprite_image,cmap='gray')
    plt.imshow(sprite_image,cmap='gray')

    ## Initialising the model parameters
    print 'Constructing the model'
    model = {}
    model['conv1'] = cnn._conv_layer(image,[5,5,1],32,'conv1')
    model['max_pool1'] = cnn._max_pool(model['conv1'],'max_pool1')
    model['conv2'] = cnn._conv_layer(model['max_pool1'],[5,5,32],64,'conv2')
    model['max_pool2'] = cnn._max_pool(model['conv2'],'max_pool2')
    model['fully_connected'] = cnn._fully_connected(model['max_pool2'],1024,'fully_connected')
    model['dropout'],keep_prob = cnn._create_dropout(model['fully_connected'],'dropout')
    model['softmax'] = cnn._create_softmax(model['dropout'],10,'softmax')
    model['loss'] = cnn._create_loss(y, model['softmax'])
    model['accuracy'] = cnn._output_accuracy(y, model['softmax'])
    model['global_step'] = tf.Variable(0,dtype=tf.int32,trainable=False,name="global_step")
    model['optimizer'] = tf.train.AdamOptimizer(LEARNING_RATE).minimize(model['loss'],global_step = model['global_step'])
    model['summary_op'] = cnn._create_summaries(model)
    print 'Model initialised....'

    ## Passing the initialised model to train() method
    ch = str(raw_input('Want to train the model: '))
    if ch == 'y' or ch == 'Y':
        
        cnn.train(model,mnist,n_training_size,image,y,keep_prob)

    ## Visualising the embeddings
    print 'Visualising the embeddings'
    placeholders = (image, y)
    visualiseEmbeddings(x_batch, y_batch, placeholders, model['fully_connected'])

if __name__ == '__main__':
    main()
