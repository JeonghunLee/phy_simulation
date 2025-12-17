# PHY Simuation 

<br/>

CXL 와 PCIe도 각 Phase Detect기반으로 생각하여, 관련부분을 DSP기반으로 다시 생각하게됨 

<br/>

## RF Mixer Test 

<br/>

* 블로그를 위해서 작성했으며, 관련설명고 각 Filter 부분 설명    
    * SDR 부분 추후 확인 진행    
    * IIR 부분은 나중에~~~ 귀찮음        

<br/>

* Mixer 기반 TEST 및 FIR Filter TEST    
    * [RF-Mixer out 비교](./notebooks/mixer_compare.ipynb)           
    * [RF-Mixer out FIR1](./notebooks/mixer_fir_step1_test.ipynb)  
    * [RF-Mixer out FIR2](./notebooks/mixer_fir_step2_comparison.ipynb)  
    * [RF-Mixer out FIR3](./notebooks/mixer_fir_step3_conv.ipynb)         

<br/>

## PHY 의 Simuation 

<br/>

* PHY(Tranceiver)를 DSP기반으로 각 부분 Simulation 하는 부분을 생각 
    * PHY TEST 
        * 여러 고속 Serial Interface PHY들 (USB,PCIe,MIPI등)   
        * Scope의 Probe 대신 RF Mixer 걸쳐 Filter  적용하는 방식으로 생각     
        * 나중에 다시 생각을 해보고, 진행   


<br/>