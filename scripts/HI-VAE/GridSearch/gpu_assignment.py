def gpu_assignment(gpus, allow_growth = True, per_process_gpu_memory_fraction = 0.95):
    """
    !! run this BEFORE importing TF or keras !!
    modified from http://kawahara.ca/select-single-gpu-keras/
    
    # Arguments
    
        (array of integers) gpus:
            [1]:   Only device 1 will be seen
            [0,1]: Devices 0 and 1 will be visible
            []:    None GPU, only CPU
            
        (boolean) allow_growth: Don't pre-allocate memory, allocate as-needed.
        
        (float) per_process_gpu_memory_fraction: determines the fraction of the overall amount of memory that each visible GPU should be allocated.
        
    # Note
        More details see https://www.tensorflow.org/guide/using_gpu
        
    # Example
        from gpu_assignment.gpu_assignment import *
        gpu_assignment([0]) # GPU device 1 will be visible, no memory preallocated (instead it'll grow as-needed).
        import tensorflow as tf
        import keras
    """
    
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    """
    Environment Variable Syntax   Results
    CUDA_VISIBLE_DEVICES=1      Only device 1 will be seen
    CUDA_VISIBLE_DEVICES=0,1    Devices 0 and 1 will be visible
    CUDA_VISIBLE_DEVICES=”0,1”  Same as above, quotation marks are optional
    CUDA_VISIBLE_DEVICES=0,2,3  Devices 0, 2, 3 will be visible; device 1 is masked
    CUDA_VISIBLE_DEVICES=""     None GPU, only CPU
    """
    
    gpus_string = ""
    for gpu in gpus:
        gpus_string += "," +str(gpu)
    gpus_string = gpus_string[1:] # drop first comma
    os.environ["CUDA_VISIBLE_DEVICES"]=gpus_string


    ###################################
    ## extra imports to set GPU options
    ###################################
    import tensorflow as tf
    #from keras import backend as k

    # TensorFlow wizardry
    config = tf.ConfigProto() 
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = allow_growth 
    # Only allow a total fraction the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction 
    # Create a session with the above options specified.
    #k.tensorflow_backend.set_session(tf.Session(config=config))
    
    return config