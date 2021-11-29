from scipy import signal
import scipy.io as sio
import streamlit as st
from datetime import date
import numpy as np
import pywt
from scipy import stats
from sklearn.utils import resample
import matplotlib.pyplot as plt
import pandas as pd
# st.set_page_config(layout="wide")
annotationFileName="annotation.csv"

st.title('FECG anomaly annotation')

def denoise(data): 
    w = pywt.Wavelet('sym4')
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    threshold = 0.07 # Threshold for filtering

    coeffs = pywt.wavedec(data, 'sym4', level=maxlev)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
        
    datarec = pywt.waverec(coeffs, 'sym4')
    
    return datarec

def prepareData():
    path = "E:\Workspaces\DopplerFHR\Data\\ninfea-non-invasive-multimodal-foetal-ecg-doppler-dataset-for-antenatal-cardiology-research-1.0.0\\dataSave.mat"
    window_size = 1000
    maximum_counting = 10000

    AllSignals = list()
    mat_contents = sio.loadmat(path)
    for i in range(24):
        AllSignals.append(mat_contents['dataSave'][0, i])
    AllSignals = np.array(AllSignals)
    # Records
    finalBeats = []
    for r in range(24):
        signals = AllSignals[r][:, 0]
        # Plot an example to the signals
        # if r is 1:
        #     # Plot each patient's signal
        #     plt.title(str(r) + " Wave")
        #     plt.plot(signals[0:700])
        #     plt.show()

        signals = denoise(signals)
        # Plot an example to the signals
        # if r is 1:
        #     # Plot each patient's signal
        #     plt.title(str(r) + " wave after denoised")
        #     plt.plot(signals[0:700])
        #     plt.show()

        signals = stats.zscore(signals)
        # Plot an example to the signals
        # if r is 1:
        #     # Plot each patient's signal
        #     plt.title(str(r) + " wave after z-score normalization ")
        #     plt.plot(signals[0:700])
        #     plt.show()
        beates = []
        for beatNumber in range(500):
            beates.append(signals[beatNumber*window_size:beatNumber*window_size+window_size])
        beates = np.array(beates)
        # st.write(beates.shape)
        try:
            signals = signal.resample(beates, 360*3, axis=1)
            finalBeats.append(signals)
        except:
            pass
    finalBeats = np.asarray(finalBeats)
    finalBeatsReshaped = finalBeats.reshape((-1, 360*3, 1))
    return finalBeatsReshaped
finalBeatsReshaped = prepareData()

def initState(finalBeatsReshaped):
    if 'annotaion' not in st.session_state:
        df=pd.DataFrame({"anomaly":np.zeros([finalBeatsReshaped.shape[0],1])[:,0]})
        df.to_csv(annotationFileName,index=False)
        st.session_state['annotaion']= 'created'

def registerAnnotation(user_input,time):
    df=pd.read_csv(annotationFileName)
    df['anomaly'].iloc[time]=user_input
    df.to_csv(annotationFileName,index=False)
    st.success("registered {}".format("Normal" if user_input==0  else "Abnroaml"))
    
def showAnnotationTool(finalBeatsReshaped):
    plot = st.empty()
    time = st.slider('which epoch?', 0, 1500)
    epochLabel = st.radio("How is this epoch?",('Normal', 'Abnormal'),index=0)
    saveBtnClicked = st.button("Save")
 
        
    if saveBtnClicked:
        if epochLabel=="Normal":
            user_input=0.0
            registerAnnotation(user_input,time)
        elif epochLabel=="Abnormal":
            user_input=1.0
            registerAnnotation(user_input,time)
            
    df=pd.read_csv(annotationFileName)
    st.dataframe(df) 

    plot_signal(finalBeatsReshaped[time],plot,time)

     
def plot_signal(data,image_place_holder,time):
    fig = plt.figure(figsize=(12,6))
    plt.plot(data)
    plt.title("epoch number: {}".format(time))
    plt.grid()
    image_place_holder.write(fig)


def app():
    initState(finalBeatsReshaped)
    showAnnotationTool(finalBeatsReshaped)

if __name__=='__main__':
    app()