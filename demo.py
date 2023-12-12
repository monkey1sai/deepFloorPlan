import os
import argparse
import numpy as np
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
from matplotlib import image
#from matplotlib.pyplot import imread
from skimage.transform import resize
from skimage.io import imread

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# input image path
parser = argparse.ArgumentParser()

parser.add_argument('--im_path', type=str, default='./demo/demo1.jpg',
                    help='input image paths.')

# color map
floorplan_map = {
	0: [255,255,255], # background
	1: [192,192,224], # closet
	2: [192,255,255], # batchroom/washroom
	3: [224,255,192], # livingroom/kitchen/dining room
	4: [255,224,128], # bedroom
	5: [255,160, 96], # hall
	6: [255,224,224], # balcony
	7: [255,255,255], # not used
	8: [255,255,255], # not used
	9: [255, 60,128], # door & window
	10:[  0,  0,  0]  # wall
}

def ind2rgb(ind_im, color_map=floorplan_map):
	rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

	for i, rgb in color_map.items():
		rgb_im[(ind_im==i)] = rgb

	return rgb_im

def tf1_to_tf2():
	with tf.compat.v1.Session() as sess:
		sess.run(tf.compat.v1.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()))
		# restore pretrained model
		saver = tf.compat.v1.train.import_meta_graph('./pretrained/pretrained_r3d.meta')
		saver.restore(sess, './pretrained/pretrained_r3d')
		# get default graph
		graph = tf.compat.v1.get_default_graph()
		# save model as *.pb
		tf.compat.v1.train.write_graph(graph, './pretrained/model', 'pretrained_r3d.pb', as_text=True)
		saver.save(sess, './pretrained/model/checkpoint.ckpt')


def wrap_frozen_graph(graph_def, inputs, outputs):
	def _imports_graph_def():
		tf.compat.v1.import_graph_def(graph_def, name="")

	wraped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
	import_graph = wraped_import.graph
	return wraped_import.prune(
		tf.nest.map_structure(import_graph.as_graph_element, inputs),
		tf.nest.map_structure(import_graph.as_graph_element, outputs)

	)

def loadtf2(filepath):
	with tf.io.gfile.GFile(filepath, 'rb') as f:	
		graph_def = tf.compat.v1.GraphDef()
		loaded = graph_def.ParseFromString(f.read())

	frozen_func = wrap_frozen_graph(
		graph_def= graph_def,
		inputs="inputs:0",
		outputs=['Cast:0', 'Cast_1:0']
	)
	return frozen_func
 

def tf2Detect(im_path):
	# load tf2 model
	model = loadtf2('./pretrained/model/pretrained_r3d.pb')

	# load image
	im = imread(im_path)
	im = im.astype(np.float32)
	im = resize(im, (512,512,3)) / 255.
	outpath = im_path.replace('.jpg', '') + 'out.jpg'
	colorarea = image.imread("colorarea.jpg")


	# infer results
	[room_type, room_boundary] = model(im)
	room_type, room_boundary = np.squeeze(room_type), np.squeeze(room_boundary)

	# merge results
	floorplan = room_type.copy()
	floorplan[room_boundary==1] = 9
	floorplan[room_boundary==2] = 10
	floorplan_rgb = ind2rgb(floorplan)

	# plot results
	plt.subplot(131)
	plt.imshow(im)
	plt.subplot(132)
	plt.imshow(floorplan_rgb/255.)
	image.imsave(outpath, floorplan_rgb/255.)
	plt.subplot(133)
	plt.imshow(colorarea)

	plt.show()



def main(args):
	# load input
	im = imread(args.im_path)
	im = im.astype(np.float32)
	im = resize(im, (512,512,3)) / 255.

	colorarea = image.imread("colorarea.jpg")

	outpath = args.im_path.replace('.jpg', '') + 'out.jpg'
	# create tensorflow session
	with tf.compat.v1.Session() as sess:
		
		# initialize
		#sess.run(tf.com.group(tf.global_variables_initializer(),
		#			tf.local_variables_initializer()))
		
		sess.run(tf.compat.v1.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()))

		# restore pretrained model
		saver = tf.compat.v1.train.import_meta_graph('./pretrained/pretrained_r3d.meta')
		saver.restore(sess, './pretrained/pretrained_r3d')

		# get default graph
		graph = tf.compat.v1.get_default_graph()

		
		# save model as *.pb
		tf.compat.v1.train.write_graph(graph, './pretrained/model', 'pretrained_r3d.pb', as_text=True)
		saver.save(sess, './pretrained/model/checkpoint.ckpt')
		
		
		# restore inputs & outpus tensor
		x = graph.get_tensor_by_name('inputs:0')

		room_type_logit = graph.get_tensor_by_name('Cast:0')
		room_boundary_logit = graph.get_tensor_by_name('Cast_1:0')

		# infer results
		[room_type, room_boundary] = sess.run([room_type_logit, room_boundary_logit],\
										feed_dict={x:im.reshape(1,512,512,3)})
		room_type, room_boundary = np.squeeze(room_type), np.squeeze(room_boundary)

		# merge results
		floorplan = room_type.copy()
		floorplan[room_boundary==1] = 9
		floorplan[room_boundary==2] = 10
		floorplan_rgb = ind2rgb(floorplan)

		# plot results
		plt.subplot(131)
		plt.imshow(im)
		plt.subplot(132)
		plt.imshow(floorplan_rgb/255.)
		image.imsave(outpath, floorplan_rgb/255.)
		plt.subplot(133)
		plt.imshow(colorarea)

		plt.show()
		

def detect(im_path):
	# load input
	im = imread(im_path)
	im = im.astype(np.float32)
	im = resize(im, (512,512,3)) / 255.

	colorarea = image.imread("colorarea.jpg")

	outpath = im_path.replace('.jpg', '') + 'out.jpg'


	# create tensorflow session
	with tf.compat.v1.Session() as sess:
		
		# initialize
		#sess.run(tf.com.group(tf.global_variables_initializer(),
		#			tf.local_variables_initializer()))
		
		sess.run(tf.compat.v1.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()))

		# restore pretrained model
		saver = tf.compat.v1.train.import_meta_graph('./pretrained/pretrained_r3d.meta')
		saver.restore(sess, './pretrained/pretrained_r3d')

		# tf.saved_model(sess, './pretrained/session_pretrained_r3d')
		# 

		# get default graph
		graph = tf.compat.v1.get_default_graph()

		# restore inputs & outpus tensor
		x = graph.get_tensor_by_name('inputs:0')

		room_type_logit = graph.get_tensor_by_name('Cast:0')
		room_boundary_logit = graph.get_tensor_by_name('Cast_1:0')

		# infer results
		[room_type, room_boundary] = sess.run([room_type_logit, room_boundary_logit],\
										feed_dict={x:im.reshape(1,512,512,3)})
		room_type, room_boundary = np.squeeze(room_type), np.squeeze(room_boundary)

		# merge results
		floorplan = room_type.copy()
		floorplan[room_boundary==1] = 9
		floorplan[room_boundary==2] = 10
		floorplan_rgb = ind2rgb(floorplan)

		# plot results
		plt.subplot(131)
		plt.imshow(im)
		plt.subplot(132)
		plt.imshow(floorplan_rgb/255.)
		image.imsave(outpath, floorplan_rgb/255.)
		print(f'save floorplan {outpath}')
		plt.subplot(133)
		plt.imshow(colorarea)
		#plt.show()




if __name__ == '__main__':

	tf1_to_tf2()

	#tf2Detect('./demo/1.jpg')
	
	# FLAGS, unparsed = parser.parse_known_args()
	# path = FLAGS.im_path
	# if os.path.isdir(path):
	# 	for filename in os.listdir(path):
	# 		if filename.count("out") > 0:
	# 			continue
	# 		if filename.endswith('.jpg') or filename.endswith('.jpeg') :
	# 			file_path = os.path.join(path, filename)
	# 			detect(file_path)
	# else:
	# 	main(FLAGS)