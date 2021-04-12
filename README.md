# NSMC_Classification


# Data
All datas are from https://github.com/e9t/nsmc

# Goal
Sentiment Classification: Classify Good/Bad from movie reviews

# ACC
###  - CNN 82.5% (Using data 100%, epochs=20)
> ####  Yoon Kim. Convolutional Neural Networks for Sentence Classification. arXiv:1408.5882
> ####   CNN(Convolution Neural Networks) 필터를 통한 문장 의미 분석 방법이 Classification tasks에 효과적인 성능 보였다.
### - (multilingual)BART 70% (Using data 10%, epochs=1)
> ####  Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, Luke Zettlemoyer. BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. arXiv:1910.13461
> #### BART는 input 문서 일부에 noise를 추가하여 다시 원문을 복구하는 AutoEncoder 형태로 학습 시킨 Pre-trained model <br> Bidirectional(BERT) 인코더와 Auto-Regressive Transformers(GPT) 디코더를 결합한 구조 <br> 임의의 길이만큼 input 문서를 마스킹 해서 마스킹 된 부분을 예측하고 전체 문장 길이에 대한 추론을 강요하는 Text Infilling 마스킹 기법을 사용했다.
### - (multilingual)BERT 81% (Using data 10%, epochs=1)
> #### Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018)
> #### BERT는 Transformer Encoder를 적층시켜 NSP(Next Sentence Prediction) 과 MLM(Masked Language Model) Task를 학습 시킨 Pre-trained model 
###  - KoELECTRA 84.1% (Using data 10%, epochs=1)
> #### 공부중입니다.
###  - KoBERT 80% (Using data 10%, epochs=1)
> https://github.com/SKTBrain/KoBERT
