"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf

import vgg19_fastmal as vgg19
import utils

import argparse 
import os
import csv
import numpy as np

#python omero_vgg19_trainable_tbnails3.py --dataset ../mp_segmented/Train/ --csv_labels ../mp_initialData/slides_labelsd2d4.csv --test_dir ../mp_segmented/Test/ --test_csv_labels ../mp_f1Data/f1_slide_labels_cleand2d4.csv --output_dir ../mcnn_segment_output/
parser = argparse.ArgumentParser(description='FastMal Classification')
#parser.add_argument('--pred_json', dest='pred_json', default='355.json', help='json file containing the prediction output of the OmeroRFCN model')
#parser.add_argument('--gt_json', dest='gt_json', default='355.json', help='json file containing the gorund truth annotations')
parser.add_argument('--dataset', dest='dataset', default='.', help='path to slide images')
parser.add_argument('--dataset2', dest='dataset2', default=None, help='path to slide images')
parser.add_argument('--csv_labels', dest='csv_labels', default='.', help='the labels per slide in csv format')
parser.add_argument('--save_dir', dest='save_dir', default='../sickle_fastmal_models_march', help='path to save the trained model')
parser.add_argument('--test_dir', dest='test_dir', default='.', help='path to the test folder')
parser.add_argument('--test_dir2', dest='test_dir2', default=None, help='path to the test folder')
parser.add_argument('--test_csv_labels', dest='test_csv_labels', default='.', help='path to the test folder')
parser.add_argument('--output_dir', dest='output_dir', default='.', help='path to the test folder')

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] ='1'#str(args.gpu)

IMSIZE = 128
num_labels=2
num_steps = 20001
batch_size=1
rpt_interval=100
min_nb_images = 1


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
# load folder names
# load labe csv file

with open(args.csv_labels, newline='') as csvfile:
    data = np.array(list(csv.reader(csvfile)))
print(data)
csv_slide_ids = data[:,0]
csv_labels = np.array(data[:,1]).astype(np.uint8)
onehot_labels = np.zeros(shape=(csv_labels.shape[0], num_labels), dtype=np.float32)
onehot_labels[csv_labels==0,0]=1
onehot_labels[csv_labels==1,1]=1
#print(csv_slide_ids, csv_labels, onehot_labels)
#slides = utils.load_folder('/home/pmanescu/OMERO/testGit/FASt-Mal-IMS/omero/vagrant/TestOmeroScripts/trainFovSlides/111017-02/')


with open(args.test_csv_labels, newline='') as csvfile:
    test_data = np.array(list(csv.reader(csvfile)))

test_csv_slide_ids = test_data[:,0]
test_csv_labels = np.array(test_data[:,1]).astype(np.uint8)



selected_slides = []
selected_labels=[]
selected_onehot_labels = []

test_slides = []
test_labels=[]
test_onehot_labels = []

subdirs = [x[0] for x in os.walk(args.dataset)][1:]  
subdirs2=[]
if args.dataset2 is not None:   
    subdirs2 = [x[0] for x in os.walk(args.dataset2)][1:]     

subdirsAll=subdirs+subdirs2
test_subdirs = [y[0] for y in os.walk(args.test_dir)]#[1:]
#print(test_subdirs)
test_subdirs2=[]
if args.test_dir2 is not None:   
    test_subdirs2 = [y[0] for y in os.walk(args.test_dir2)]#[1:]
     
test_subdirsAll=test_subdirs+test_subdirs2

#                                                                       
for subdir in subdirsAll: 
    predicted_rois = []
    slide_path = subdir
    slide_id = slide_path.split(os.path.sep)[-1]
    if slide_id in csv_slide_ids:
        sel_id = list(csv_slide_ids).index(slide_id)
        print(slide_id, ' has label ', csv_labels[sel_id], ' or ', onehot_labels[sel_id,:])
        selected_slides.append(slide_path) 
        selected_onehot_labels.append(onehot_labels[sel_id,:])
        selected_labels.append(csv_labels[sel_id])
    #dataset_predictions={"dataset_id":dataset_id}
    
    
for tsubdir in test_subdirsAll: 
    #predicted_rois = []
    slide_path = tsubdir#'/home/petre/mount_point/validation_data/355/'
    
    #print (slide_path.split(os.path.sep))
    #dataset_id = 
    
    slide_id = slide_path.split(os.path.sep)[-1]
    #print(slide_id)
    if slide_id in test_csv_slide_ids:
        sel_id = list(test_csv_slide_ids).index(slide_id)
        print(slide_id, ' has label ', test_csv_labels[sel_id])
        test_slides.append(slide_path) 
        #test_onehot_labels.append(onehot_labels[sel_id,:])
        test_labels.append(test_csv_labels[sel_id])    


selected_labels=np.array(selected_labels)
positive_ids = np.nonzero(selected_labels)[0]
negative_ids = np.nonzero(1-selected_labels)[0]


print("#Negative train:", np.sum(1-selected_labels))
print("#Positive train:", np.sum(selected_labels))

print("#Negative test:", np.sum(1-np.array(test_labels)))
print("#Positive test:", np.sum(np.array(test_labels)))

      
#print(slides.shape)
with tf.Session() as sess:

    #saver = tf.train.Saver()
    images = tf.placeholder(tf.float32, [None, IMSIZE, IMSIZE, 3])
    true_out = tf.placeholder(tf.float32, [None, 2])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg19.Vgg19('../models/vgg19.npy', imsize=128)
    #vgg = vgg19.Vgg19('../sickle_fastmal_models_march/sickle_classifier1103_max_segmentation_ce15000_vgg19_model.npy', imsize=128)
    #vgg=vgg19.Vgg19('../focal_loss_models/malaria_classifier2911_mean_retinanet_ce_loss_d2d450000_vgg19_model.npy',imsize=64)
    #vgg.build_loop_tbnails(images, no_images=10, train_mode=train_mode)
    vgg.build_avg_pool(images, train_mode=train_mode)
    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    #print(vgg.get_var_count())



    # test classification
    #prob= sess.run(vgg.prob,  feed_dict={images: batch1, train_mode: False})
    
    #test_fc =sess.run(vgg.new_prob,  feed_dict={images: slides, train_mode: False})
    #utils.print_prob(prob[0], './synset.txt')
    #print(test_fc)
    # simple 1-step training
    #cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
    #w=[1.0, 1.25]
    #weighted_logits = tf.multiply(vgg.new_fc8, w)    
    focal_loss = tf.reduce_mean(
     tf.nn.softmax_cross_entropy_with_logits_v2(logits=vgg.new_fc8, labels=true_out)) 
    
    #focal_loss  = utils.focal_loss_softmax(true_out,vgg.new_fc8, gamma=1)
    train = tf.train.GradientDescentOptimizer(0.0003).minimize(focal_loss)
    #train = tf.train.AdamOptimizer( 0.001).minimize(focal_loss)
    sess.run(tf.global_variables_initializer())
    #sess.run(tf.local_variables_initializer())    
    step_pos=0
    step_neg=0
    for step in range(1, num_steps+1):
      try:  
          offset = (step * batch_size) % (len(selected_onehot_labels) - batch_size)
          
          if step%2==0:
              sl_id = positive_ids[step_pos % (positive_ids.shape[0])]
              step_pos=step_pos+1
          else:
              sl_id = negative_ids[step_neg % (negative_ids.shape[0])]
              step_neg=step_neg+1

          #print ( selected_slides[sl_id])
          slides = utils.load_folder_random(selected_slides[sl_id], max_no_img=100, crop_size=128)
        
          #print (len(slides))
          if len(slides)>min_nb_images:
              slides = np.array(slides)
              #print(slides.shape)
              labels = np.reshape(selected_onehot_labels[sl_id], (batch_size, num_labels))
                #train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
               #sess.run(train, feed_dict={images: batch1, true_out: [img1_true_result], train_mode: True})
              _,l = sess.run([train, focal_loss], feed_dict={images: slides, true_out: labels, train_mode: True})
          if (step % rpt_interval == 0):
              print('Minibatch loss at step %d: %f' % (step, l))     
          if (step % 5000 == 0):   
            save_path = os.path.join(args.save_dir,'sickle_classifier1503_mean_segmentation_ce'+str(step)+"_vgg19_model.npy")
            vgg.save_npy(sess, save_path)
            print("Model saved in file: %s" % save_path)    
      except IOError as e:
        print('Could not read:', selected_slides[offset], ':', e, '- it\'s ok, skipping.')        
# test classification again, should have a higher probability about tiger
    #prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
    #utils.print_prob(prob[0], './synset.txt')

    # test save
    #vgg.save_npy(sess, './test-save.npy')
    prediction_csv = 'sickle_prediction_mean_segmentation1503ce.csv' 
    header=['Slide-Id', 'True', 'Predicted']
    predictionFile= open(os.path.join(args.output_dir, prediction_csv),'w')  
    wr = csv.writer(predictionFile, dialect='excel')
    wr.writerow(header)    
    predicted_classif = np.zeros(len(test_slides))
    true_classif = np.zeros(len(test_slides))
    for tt in range(len(test_slides)):
        tslides = utils.load_folder_random(test_slides[tt], max_no_img=500,crop_size=128)
        
        tslide_id = test_slides[tt].split(os.path.sep)[-1]
        malaria_classif=0
        if len(tslides)>min_nb_images:
          tslides = np.array(tslides)
          #print(tslides.shape)
          #tlabels = np.reshape(test_onehot_labels[tt], (batch_size, num_labels))
          prob = sess.run(vgg.new_prob, feed_dict={images: tslides, train_mode: False})
          predicted_classif[tt] = np.argmax(prob)
        sel_id = list(test_csv_slide_ids).index(tslide_id)
        true_classif[tt] = test_csv_labels[sel_id]
        wr.writerow([tslide_id, true_classif[tt], predicted_classif[tt]])     
    

    overall_accuracy=np.mean(true_classif==predicted_classif)    
    print('Overall accuracy', overall_accuracy)
    true_pos = true_classif[true_classif==1]
    pred_pos = predicted_classif[true_classif==1]
    positive_accuracy=np.mean(true_pos==pred_pos)
    print('Postive accuracy', positive_accuracy)
    true_neg = true_classif[true_classif==0]    
    pred_neg = predicted_classif[true_classif==0]
    negative_accuracy=np.mean(true_neg==pred_neg)
    print('Negative accuracy', negative_accuracy)    
        

