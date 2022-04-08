
#설치
import nltk
import scipy
nltk.download('stopwords')

#영어 불용어 리스트 확인
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
print(stopwords.words('english'))

#txt에서 불용어 삭제하는 for문
example= "# 예시문장 "
stop_words=set(stopwords.words('english'))

word_tokens = word_tokenize(example)

result=[]
for w in word_tokens:
    if w not in stop_words:
        result.append(w)

print(word_tokens, '\n')
print(result)

#___________________________불용어 제거 실습 V.eng__________________________


#------------------------------한국어 형태소 토큰화---------------------------------
#예시문장
data="치킨전문점에서|고객의주문에의해|치킨판매|산업공구|다른 소매업자에게|철물 수공구|절에서|신도을 대상으로|불교단체운영|영업장에서|고객요구로|자동차튜닝|실내포장마차에서|접객시설을 갖추고|소주,맥주제공|철,아크릴,포맥스|스크린인쇄|명판"

#okt (open korea text) (구 Twitter)
from konlpy.tag import Okt
okt=Okt()
print(okt.morphs(data))     #morphs: 형태소 추출    #가장 기본적이고 단순한 형태
print(okt.pos(data))        #pos: 품사 태깅
print(okt.nouns(data))      #nouns: 명사추출     #? 포맥스 -> 포/ 맥스로 분리됨 - 어떻게 설정할 수 있지?

#꼬꼬마(kkma)
from konlpy.tag import Kkma
kkma=Kkma()
print(kkma.morphs(data))     #morphs: 형태소 추출
print(kkma.pos(data))        #pos: 품사 태깅
print(kkma.nouns(data))      #nouns: 명사추출  #치킨/ 치킨전문점/ 전문점 <- 구체적으로 나뉨 다만 중복이 많아 고려 필요

#메캅(Mecab) -  버려   / 상대적으로 빠르다고 함
from konlpy.tag import Mecab
#mecab=Mecab()

#코모란(Komoran)                #불교를 싫어하는 것으로 추정됨 / 인식하지 못하는 문장이 있음.
from konlpy.tag import Komoran
komo=Komoran()
print(komo.morphs(data))     #morphs: 형태소 추출
print(komo.pos(data))        #pos: 품사 태깅
print(komo.nouns(data))      #nouns: 명사추출

#한나눔(Hannanum)               # 개쓰레기임 / 왜만든건지 1도 이해할 수 없음
from konlpy.tag import Hannanum
hanna=Hannanum()
print(hanna.morphs(data))     #morphs: 형태소 추출
print(hanna.pos(data))        #pos: 품사 태깅
print(hanna.nouns(data))      #nouns: 명사추출 

# 2022.03.29
# 미션 1: 꼬꼬마, okt 중 고유명사로 인식하는 것을 어떻게 바꾸는지.
# 미션 2: 자체적 불용어 리스트 만들기
# 미션 3: 나이브 베이지안 적합 방향 생각하기.. 공부도 좀 하기 제발,,,,
# 미션 4: 나이브 베이지안 외 적용 가능한 분류 모델 개념 찾고 개념 챙겨오기,,,


#------------------------------------------------------------------
# 2022.04.01 보리 생일 >.<
#  오늘의 과제 1: 테이블에서 토큰화 방법

#데이터 불러오기
import pandas as pd
import sklearn

    ## test set과 train set 분리하기 (2:8)
df = pd.read_table('E:/가천대/AI 활용대회/파이썬/실습용자료.txt',sep='|',encoding = 'euc-kr')
df.head()
df.info()
#import sklearn
from sklearn.model_selection import train_test_split


train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
len(test_set)
len(train_set)
    ## train_set은 'dset'으로 명명 
dset=pd.DataFrame(train_set)


#병합할 데이터 고르기
kordata = dset.loc[:,['text_obj', 'text_mthd', 'text_deal']]
kord=['text_obj','text_mthd','text_deal']

dset[kord]=dset[kord].astype(str)

# str 데이터 병합하기
# kordata['kor'] = kordata[kord].apply(lambda row:' '.join(row.values.astype(str), axis=1)) #이게 안되는데 
kordata['kor'] = kordata['text_obj'] +' '+ kordata['text_mthd'] +' '+ kordata['text_deal']  #이게 된다고? 시발?

kordata.head()
kordata['kor']=kordata['kor'].astype(str)
kordata.info()

#-------------------------------------------------------------------------------
#2022.04.05 
#정리해야할 순서
# 1. x와 y 분리하기
# 2. (논의 필요) ['text_obj', 'text_mthd', 'text_deal']을 합쳐야 하는가 /합치는게 맞는듯 띄어쓰기 문제때문/
# 3. test set과 train set 분리하기
# 4. 토큰화 전 PYKoSpacing 이용하여 띄어쓰기 맞게 바꾸기

## 미션 
#  단어 토큰화 패키지 선정하기

# www.https://wikidocs.net/92961
# soynlp 추천?
# customized konlpy <- 사용자 사전 추가 용이

# x와 y 분리하기
dset.head()
dset['kor'] = kordata['kor']

x = dset['kor']
y= dset[['digit_1','digit_2','digit_3']]

x1 = list(dset.loc[:,'kor'])
y1 = list(dset.loc[:,'digit_1'])
y2 = list(dset.loc[:,'digit_2'])
y3 = list(dset.loc[:,'digit_3'])


# print(x) 중지는 ctrl + C

#x,y 분할 후 test set 분리
x_train_set, x_test_set, y_train_set, y_test_set = train_test_split(x,y, test_size=0.2, random_state=42)
len(x_test_set)
len(x_train_set)
len(y_test_set)
len(y_train_set)



#------------------------------------------22.04.07------------------------
   ## 데이터 불러오기
df = pd.read_table('E:/가천대/AI 활용대회/파이썬/실습용자료.txt',sep='|',encoding = 'euc-kr')
df.head()
df.info()
#import sklear
from sklearn.model_selection import train_test_split

    ##한글 데이터 병합
kordata = df.loc[:,['text_obj', 'text_mthd', 'text_deal']]
df1=df
kordf = df1['text_obj'] +' '+ df1['text_mthd'] +' '+ df1['text_deal'] 
df1['kor']=kordf.astype(str)

kordata.head()


    ## x와 y 분리하기

x = df1['kor']
y = df1[['digit_1','digit_2','digit_3']]



    ## pykospacing  띄어쓰기 도전
from pykospacing import Spacing

# 공백 없애는 코드
def remove_blank(x):
    return x.replace(' ', '')

ac = df['kor'].apply(remove_blank)
df1['kor'].head()
ac.head()

spacing=Spacing()
a = spacing(ac)
a.head()
a[1]

# 184번쨰 코드 돌아가는지 확인 바람.


#04.08
import re
import pandas as pd
from tqdm import tqdm
from konlpy.tag import Okt
from pykospacing import Spacing
from collections import Counter

a = x.apply(remove_blank)
a

def extract_word(text):
    hangul = re.compile('[^가-힣]') 
    result = hangul.sub(' ', text) 
    return result

extract_word(df['kor'])

lix = df['kor'].values.tolist()
type(lix)
lix[1]
s = []

for i in range(0,len(lix)):
    s.append(extract_word(lix[i]))

#빈칸 없애기
def remove_blank(x):
    return x.replace(' ', '')

for i in range(0,len(s)):
    s[i] = remove_blank(s[i])

spacing=Spacing()
for i in range(0,len(s)):
    s[i] = spacing(s[i])

s[1]


