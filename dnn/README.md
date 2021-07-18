# Deep Learning Model

Proof of Concept for Distributed Matrix Multiplication in Deep Learning Model. The code uses the decoding probability of the UEP codes based on the packet arrival count to erase packets of the classes that did not arrive. For the other codes, it erases the packets that did not arrive. The resulatant packets are pre-computed before erasure.

The deep learning layers are built from scratch using both numpy and scipy. Keras library is called only to access the dataset. 

# Requirements
pip install tqdm

pip install keras

pip install scikit-image
