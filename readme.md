This package contains the code and pre-trained weights of the experiments reported in the manuscript "Signal retrieval with measurement system knowledge using variational generative model". The folder include 3 signal retrieval examples, ultrafast Pulse retrieval, Fresnel in-line hologram and video compressive sensing. The main executable files associated with each experiment are:

main_pulse.py (ultrafast pulse retrieval)
main_hologram.py (Fresnel in-line hologram)
main_compression.py (video compressive sensing)

The pre-trained weights are stored in ./model/. To load pre-trained weights, set 'load_model=True' in the main executable, the program will skip the training process and load weights specified by the 'load_model_path'. To train the package, set 'load_model=False', the program will train the model and save the weights in the 'save_model_path'.

The forward model A() and encoder/decoder structures are stored separately in each signal retrieval directory as forward_model.py and layers.py.

This package has been tested under Python 3.5 and Tensorflow 1.9.0 environment.

Link to download training dataset and pre-trained weights for the Fresnel in-line hologram example:

https://knightsucfedu39751-my.sharepoint.com/:u:/g/personal/zyzhu_knights_ucf_edu/ESh8SVnbHqxItiqkrMd4PnsB11i3wDZBp6Sv-vClBaEbMA?e=mqdA41

To run the hologram example, download the files and place the dataset files in ./Hologram/dataset, and pre-trained weight files in ./model
