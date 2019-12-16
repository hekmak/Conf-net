import tensorflow as tf
import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
    
IMAGE_WIDTH, IMAGE_HEIGHT = 1216, 352
purge_thresh = 351 # in millimeter
# convert thresh between 0-65535
purge_thresh = (purge_thresh*256)/1000

out_pathes = ['/notebooks/project/predictions/error',
              '/notebooks/project/predictions/error_vis',
              '/notebooks/project/predictions/error_vis_log',
              '/notebooks/project/predictions/depth',
              '/notebooks/project/predictions/depth_vis',
              '/notebooks/project/predictions/purged'] 


os.environ["CUDA_VISIBLE_DEVICES"]="0"
height, width, channels = 352, 1216, 1
raw_dir = '/notebooks/project/sample_lidar_scans'
X_f = sorted(os.listdir(raw_dir))

model_path = "/notebooks/project/trained_model"
saver = tf.train.import_meta_graph(os.path.join(model_path,'model.ckpt-52909.meta'),clear_devices=True)

config=tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True


with tf.Session(config=config) as sess:
    graph = tf.get_default_graph()
    logits_depth = graph.get_tensor_by_name("SparseConvNet_1/ham_easy:0")
    logits_error = graph.get_tensor_by_name("SparseConvNet_1/ham_hard:0")
    saver.restore(sess,tf.train.latest_checkpoint(model_path))
    features = tf.placeholder("float", shape=[1, 352,1216,1],name='ham_input')

    i=5
    for image_num in X_f[:]:
        X_dir = os.path.join(raw_dir,image_num)
        X_png = np.array(Image.open(X_dir))
        cv_img = cv2.resize(X_png,dsize=(1216,352),interpolation=cv2.INTER_NEAREST)
        X_png = cv_img[np.newaxis,:,:,np.newaxis]
        X_batch = X_png
        X_batch = np.repeat(X_batch, 1,axis=0)
        pred_depth, pred_error = sess.run ( (logits_depth,logits_error) ,feed_dict={features:X_batch})

        # 900mm lowest acceptable depth
        lowest = (900.0*256/1000)
        pred_depth[pred_depth<lowest]=lowest
        pred_error[pred_error<0]=0


        for path in out_pathes:
          if not os.path.exists(path):
            os.makedirs(path)

        fname_v= os.path.join('/notebooks/project/predictions/error',image_num)
        fname= os.path.join('/notebooks/project/predictions/depth',image_num)
        fname_v_vis= os.path.join('/notebooks/project/predictions/error_vis',image_num)
        fname_v_vis_log= os.path.join('/notebooks/project/predictions/error_vis_log',image_num)
        fname_vis= os.path.join('/notebooks/project/predictions/depth_vis',image_num)
        fname_purged= os.path.join('/notebooks/project/predictions/purged',image_num)



        purged_image = np.copy(pred_depth)
        purged_image[pred_error>purge_thresh] = 0
        purged_image = np.squeeze(purged_image)
        print purged_image.shape
        pred_depth_image = np.squeeze(pred_depth[0,:,:,0])
        pred_error_image = np.squeeze(pred_error[0,:,:,0])
        pred_depth_image = pred_depth_image.astype(np.uint16)
        pred_error_image = pred_error_image.astype(np.uint16)
        cv2.imwrite(fname,pred_depth_image)
        cv2.imwrite(fname_v,pred_error_image)
        plt.imsave(arr= pred_error_image, fname= fname_v_vis , cmap ='nipy_spectral')
        plt.imsave(arr= np.log(pred_error_image), fname= fname_v_vis_log , cmap ='nipy_spectral')
        plt.imsave(arr= pred_depth_image, fname= fname_vis , cmap ='nipy_spectral')
        plt.imsave(arr= purged_image[0,:,:], fname= fname_purged , cmap ='nipy_spectral')

        i=i+1
        print i
