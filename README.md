# YT2Audio

### 깔 때..

python은 버전 한 3.10 ? ~ 3.11 즈음 에서 돌아가는 것 같음.

MFA 설치는 [링크](https://linguisting.tistory.com/107) 참고. <br/>
그리고 `pip install joblib==1.3.0` 해줘야 MFA 오류 안 뜸. pypi install 하지말고 conda로 install. mamba 쓸 수 있으면 써도 괜찮. <br/>


현재 다른 python file들 각각 step 별로 구현해서 진행 했으나, (split, mfa, ..) 방식을 수정함에 따라 `integrated_pipeline` 이거 하나로 다 되도록 수정. <br/>


### 간단한.. pipline

1. 1시간 전체 오디오에서 첫 5분 구간 추출
2. 추출된 5분 오디오와 1시간 전체 오디오 모두 elevenlaabs api 수행(==transcribe & speaker diarization)
3. 일반 전사본 (연속된 단어 텍스트), 화자 분리 전사본 (형식: speaker_{i}: 텍스트, 턴마다 개행) 턴 기반 구조의 json 저장
4. 추출된 5분 오디오에서 **기준 턴** 3 턴을 잡고 .. **이어서 자세하게 작성 필요** .. 5분 오디오에 대한 transcription을 기준 턴 직전까지 Full audio transcription으로 텍스트 및 턴 대체
5. MFA로 5분 오디오에 대해서 timestamp aligning, 1시간 full audio의 전사 텍스트에도 기준턴 직전까지 timestamp 정보 넘겨줌.
6. 기준턴의 직전턴의 end_time ~ 이후 5분 구간 추출
7. full audio 다 될 때 까지 Step 2~6 반복
8. 마지막에 chunk [last turn end_time : end_time + 5 min]이 마지막 5분 보다 짧으면 그냥 싹 다 대체 해서 마무리






