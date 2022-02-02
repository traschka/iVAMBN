import gpu_assignment
config=gpu_assignment.gpu_assignment([0,1,2])
import tensorflow as tf
import graph_new
import parser_arguments
import read_functions
import os
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
plt.ioff()
import time
from sklearn.model_selection import KFold

def merge_dat(lis):
    'merge all dataframes in a list on SUBJID'
    df = lis[0]
    for x in lis[1:]:
        df=pd.merge(df, x, on = 'SUBJID')
    return df

def print_loss(epoch, start_time, avg_loss, avg_loss_reg, avg_KL_s, avg_KL_z):
    'for network training output'
    print("Epoch: [%2d]  time: %4.4f, train_loglik: %.8f, KL_z: %.8f, KL_s: %.8f, ELBO: %.8f, Reg. loss: %.8f"
          % (epoch, time.time() - start_time, avg_loss, avg_KL_z, avg_KL_s, avg_loss-avg_KL_z-avg_KL_s, avg_loss_reg))

def train_network(settings,name):
    'run training (no output)'

    argvals = settings.split()
    args = parser_arguments.getArgs(argvals)
    print(args)
    #print(args.types_file)

    #Creating graph
    sess_HVAE = tf.Graph()

    with sess_HVAE.as_default():
        tf_nodes = graph_new.HVAE_graph(args.model_name, args.types_file, args.batch_size,
                                    learning_rate=args.learning_rate, z_dim=args.dim_latent_z, y_dim=args.dim_latent_y, s_dim=args.dim_latent_s,weight_decay=args.weight_decay, y_dim_partition=args.dim_latent_y_partition)

    ################### Running the VAE Training #################################
    train_data, types_dict, miss_mask, true_miss_mask, n_samples = read_functions.read_data(args.data_file, args.types_file, args.miss_file, args.true_miss_file)
    n_batches = int(np.floor(np.shape(train_data)[0]/args.batch_size))#Get an integer number of batches
    miss_mask = np.multiply(miss_mask, true_miss_mask)#Compute the real miss_mask

    with tf.Session(graph=sess_HVAE,config=config) as session:
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        print('Initizalizing Variables ...')
        tf.global_variables_initializer().run()

        start_time = time.time()
        # Training cycle
        loglik_epoch = []
        testloglik_epoch = []
        KL_s_epoch = []
        KL_z_epoch = []
        loss_epoch=[]
        for epoch in range(args.epochs):
            avg_loss = 0.
            avg_KL_s = 0.
            avg_KL_z = 0.
            avg_loss_reg=0.
            samples_list = []
            p_params_list = []
            q_params_list = []
            log_p_x_total = []
            log_p_x_missing_total = []

            # Annealing of Gumbel-Softmax parameter
            tau = np.max([1.0 - (0.999/(args.epochs-50))*epoch,1e-3])
            print(tau)

            #Randomize the data in the mini-batches
            np.random.seed(42)
            random_perm = np.random.permutation(range(np.shape(train_data)[0]))
            train_data_aux = train_data[random_perm,:]
            miss_mask_aux = miss_mask[random_perm,:]
            true_miss_mask_aux = true_miss_mask[random_perm,:]

            for i in range(n_batches):
                data_list, miss_list = read_functions.next_batch(train_data_aux, types_dict, miss_mask_aux, args.batch_size, index_batch=i) #Create inputs for the feed_dict
                data_list_observed = [data_list[i]*np.reshape(miss_list[:,i],[args.batch_size,1]) for i in range(len(data_list))] #Delete not known data (input zeros)

                #Create feed dictionary
                feedDict = {i: d for i, d in zip(tf_nodes['ground_batch'], data_list)}
                feedDict.update({i: d for i, d in zip(tf_nodes['ground_batch_observed'], data_list_observed)})
                feedDict[tf_nodes['miss_list']] = miss_list
                feedDict[tf_nodes['tau_GS']] = tau
                feedDict[tf_nodes['zcodes']] = np.ones(args.batch_size).reshape((args.batch_size,1))
                feedDict[tf_nodes['scodes']] = np.ones(args.batch_size).reshape((args.batch_size,1))

                #Running VAE
                _,loss,KL_z,KL_s,samples,log_p_x,log_p_x_missing,p_params,q_params,loss_reg  = session.run([tf_nodes['optim'], tf_nodes['loss_re'], tf_nodes['KL_z'], tf_nodes['KL_s'], tf_nodes['samples'], tf_nodes['log_p_x'], tf_nodes['log_p_x_missing'],tf_nodes['p_params'],tf_nodes['q_params'],tf_nodes['loss_reg']], feed_dict=feedDict)

                #print('shape loss'+str(loss.shape))
                #print(loss)

                #Collect all samples, distirbution parameters and logliks in lists
                samples_list.append(samples)
                p_params_list.append(p_params)
                q_params_list.append(q_params)
                log_p_x_total.append(log_p_x)
                log_p_x_missing_total.append(log_p_x_missing)

                # Compute average loss
                avg_loss += np.mean(loss)
                avg_KL_s += np.mean(KL_s)
                avg_KL_z += np.mean(KL_z)
                avg_loss_reg+=np.mean(loss_reg)

            #print('::done with batches::')
            #Concatenate samples in arrays
            s_total, z_total, y_total, est_data = read_functions.samples_concatenation(samples_list)

            #Create global dictionary of the distribution parameters
            p_params_complete = read_functions.p_distribution_params_concatenation(p_params_list, types_dict, args.dim_latent_z, args.dim_latent_s)
            q_params_complete = read_functions.q_distribution_params_concatenation(q_params_list,  args.dim_latent_z, args.dim_latent_s)

            #Compute mean and mode of our loglik models
            loglik_mean, loglik_mode = read_functions.statistics(p_params_complete['x'],types_dict)

            #Compute test-loglik from log_p_x_missing
            log_p_x_total = np.transpose(np.concatenate(log_p_x_total,1))
            log_p_x_missing_total = np.transpose(np.concatenate(log_p_x_missing_total,1))
            if args.true_miss_file:
                log_p_x_missing_total = np.multiply(log_p_x_missing_total,true_miss_mask_aux[:n_batches*args.batch_size,:])
            avg_test_loglik = np.sum(log_p_x_missing_total)/np.sum(1.0-miss_mask_aux)

            # Display logs per epoch step
            if epoch % args.display == 0:
                print_loss(epoch, start_time, avg_loss/n_batches, avg_loss_reg/n_batches, avg_KL_s/n_batches, avg_KL_z/n_batches)
                print("")
            loss_epoch.append(-avg_loss/n_batches)

        print('Training Finished ...')
        plt.clf()
        plt.figure()
        plt.plot(loss_epoch)
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction loss')  # we already handled the x-label with ax1
        plt.title(args.save_file)
        plt.savefig('output/'+args.save_file+'_'+name+'.png', bbox_inches='tight')
        return -avg_loss/n_batches


def run_network(settings,name,n_splits=3,plot=False):
    'run training (no output)'
    argvals = settings.split()
    args = parser_arguments.getArgs(argvals)
    print(name)
    print(args)

    # get full data
    data, types_dict, miss_mask, true_miss_mask, n_samples = read_functions.read_data(args.data_file, args.types_file, args.miss_file, args.true_miss_file)
    miss_mask = np.multiply(miss_mask, true_miss_mask)#Compute the real miss_mask

    # split data and run training/test per fold
    kf = KFold(n_splits=n_splits,shuffle=True)
    score_keep=[]
    fold=0
    for train_idx, test_idx in kf.split(data):
        fold += 1
        score=run_epochs(args,data,train_idx, test_idx, types_dict, miss_mask, true_miss_mask, n_samples,name+'_FOLD'+str(fold),plot)# returns final train and test score after # epochs
        score_keep.append(score[1])# keep test score
        print("Score for fold %d: Train - %.3f :: Test - %.3f" % (fold, score[0], score[1]))

    return np.mean(score_keep) # return the mean


def run_epochs(args,data,train_idx, test_idx, types_dict, miss_mask, true_miss_mask, n_samples,name,plot):
    'this creates the graph and runs train and test batches for this epoch'

    #Creating graph
    sess_HVAE = tf.Graph()
    with sess_HVAE.as_default():
        tf_nodes = graph_new.HVAE_graph(args.model_name, args.types_file, args.batch_size,
                                    learning_rate=args.learning_rate, z_dim=args.dim_latent_z, y_dim=args.dim_latent_y, s_dim=args.dim_latent_s,weight_decay=args.weight_decay, y_dim_partition=args.dim_latent_y_partition)

    n_batches_train = int(np.floor(len(train_idx)/args.batch_size))#Get an integer number of batches
    n_batches_test=int(np.floor(len(test_idx)/args.batch_size))#Get an integer number of batches

    with tf.Session(graph=sess_HVAE,config=config) as session:
        print('Initizalizing Variables ...')
        print('Train size:'+str(len(train_idx))+':: Test size:'+str(len(test_idx)))

        tf.global_variables_initializer().run()
        start_time = time.time()

        # Training cycle
        train_loss_epoch=[]
        train_KL_s_epoch = []
        train_KL_z_epoch = []
        train_loss_reg_epoch=[]
        test_loss_epoch=[]
        test_KL_s_epoch = []
        test_KL_z_epoch = []
        test_loss_reg_epoch=[]
        for epoch in range(args.epochs):
            # run inp: data,types_dict,miss_mask,true_miss_mask,n_batches,batch_size
            #losses=[avg_loss/n_batches, avg_KL_s/n_batches, avg_KL_z/n_batches, avg_loss_reg/n_batches]
            # ELBO -(avg_loss-avg_KL_z-avg_KL_s)+loss_reg

            #training
            losses_train=run_batches(session,tf_nodes,data[train_idx],types_dict,miss_mask[train_idx],true_miss_mask[train_idx],n_batches_train,args.batch_size,args.epochs,epoch,train=True)
            train_loss_epoch.append(losses_train[0])
            train_KL_s_epoch.append(losses_train[1])
            train_KL_z_epoch.append(losses_train[2])
            train_loss_reg_epoch.append(losses_train[3])

            # testing
            losses_test=run_batches(session,tf_nodes,data[test_idx],types_dict,miss_mask[test_idx],true_miss_mask[test_idx],n_batches_test,args.batch_size,args.epochs,epoch,train=False)
            test_loss_epoch.append(losses_test[0])
            test_KL_s_epoch.append(losses_test[1])
            test_KL_z_epoch.append(losses_test[2])
            test_loss_reg_epoch.append(losses_test[3])

        #fig, ax = plt.subplots(1, 3, sharex='col',figsize=(24, 8))
        #ax[0].plot(train_loss_epoch,label='train')
        #ax[0].plot(test_loss_epoch,label='test')
        #ax[0].set_ylabel('Reconstruction loss')
        #ax[0].set_xlabel('Epoch')
        #ax[1].plot(train_KL_z_epoch,label='train')
        #ax[1].plot(test_KL_z_epoch,label='test')
        #ax[1].set_ylabel('KL z loss')
        #ax[1].set_xlabel('Epoch')
        #ax[2].plot(train_loss_reg_epoch,label='train')
        #ax[2].plot(test_loss_reg_epoch,label='test')
        #ax[2].set_ylabel('Regularization loss')
        #ax[2].set_xlabel('Epoch')
        #ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #fig.suptitle(args.save_file+' '+name)
        #fig.savefig('output/'+args.save_file+'_'+name+'.png', bbox_inches='tight')
        #if plot:
        #    plt.show()
        #else:
        #    plt.close(fig)

    return [train_loss_epoch[-1],test_loss_epoch[-1]]

def run_batches(session,tf_nodes,data,types_dict,miss_mask,true_miss_mask,n_batches,batch_size,n_epochs,epoch,train):
    'This runs the batch training for a single epoch and returns performance'
    avg_loss = 0.
    avg_KL_s = 0.
    avg_KL_z = 0.
    avg_loss_reg=0.

    # Annealing of Gumbel-Softmax parameter
    tau = np.max([1.0 - (0.999/(n_epochs-50))*epoch,1e-3])

    #Randomize the data in the mini-batches
    np.random.seed(42)
    random_perm = np.random.permutation(range(np.shape(data)[0]))
    data_aux = data[random_perm,:]
    miss_mask_aux = miss_mask[random_perm,:]
    true_miss_mask_aux = true_miss_mask[random_perm,:]

    for i in range(n_batches):
        data_list, miss_list = read_functions.next_batch(data_aux, types_dict, miss_mask_aux, batch_size, index_batch=i) #Create inputs for the feed_dict
        data_list_observed = [data_list[i]*np.reshape(miss_list[:,i],[batch_size,1]) for i in range(len(data_list))] #Delete not known data (input zeros)

        #Create feed dictionary
        feedDict = {i: d for i, d in zip(tf_nodes['ground_batch'], data_list)}
        feedDict.update({i: d for i, d in zip(tf_nodes['ground_batch_observed'], data_list_observed)})
        feedDict[tf_nodes['miss_list']] = miss_list
        feedDict[tf_nodes['tau_GS']] = tau
        feedDict[tf_nodes['zcodes']] = np.ones(batch_size).reshape((batch_size,1))
        feedDict[tf_nodes['scodes']] = np.ones(batch_size).reshape((batch_size,1))

        #Running VAE
        if train:
            _,loss,KL_z,KL_s,loss_reg  = session.run([tf_nodes['optim'],tf_nodes['loss_re'], tf_nodes['KL_z'], tf_nodes['KL_s'],tf_nodes['loss_reg']], feed_dict=feedDict)
        else:
            loss,KL_z,KL_s,loss_reg  = session.run([tf_nodes['loss_re'], tf_nodes['KL_z'], tf_nodes['KL_s'],tf_nodes['loss_reg']], feed_dict=feedDict)

        # Compute average loss
        avg_loss += np.mean(loss)
        avg_KL_s += np.mean(KL_s)
        avg_KL_z += np.mean(KL_z)
        avg_loss_reg+=np.mean(loss_reg)
    return [-avg_loss/n_batches, avg_KL_s/n_batches, avg_KL_z/n_batches, avg_loss_reg/n_batches]
