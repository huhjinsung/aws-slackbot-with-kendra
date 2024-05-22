# aws-slackbot-with-kendra

## 개요
이번 레포지토리에서는 AWS의 생성형 AI 서비스인 Bedrock을 활용하여 Chatbot을 구현하며, Chatbot은 Slack의 App으로 탑재되어 기능하는 샘플 데모를 소개합니다. 생성형 AI의 LLM은 모델이 학습하지 않은 데이터 또는 최신 데이터에 대한 질문이 올 때 거짓말을 답변하는 할루시네이션 현상이 발생합니다. 이러한 할루시네이션 현상을 해결하기위해 LLM에 답변을 생성하기 이전에 질문과 관련된 데이터를 LLM에 프롬프트로 전달하여 할루시네이션을 극복하는 RAG 패턴을 사용합니다.

이번 레포지토리에서는 AWS의 Kendra 서비스를 RAG을 위한 Document 저장소로 활용하며 Langchaing 라이브러리를 통해 RAG Retreiver 기능을 사용합니다.

## 아키텍처
<img src="/Picture1.png"></img>

1. slack 사용자는 slack app을 통해 궁금한 사항을 질문합니다.
2. Slack app의 특성상 3초 안에 response를 받아야 합니다. langchain 특성상 3초 만에 답변을 생성하는 것은 불가능하기에, Slack App에게 우선 200 response를 전달하고 사용자의 질문은 SQS에 등록합니다.
3. Langchain을 구현하는 Lambda에서 SQS에서 메세지를 가져옵니다.
4. 해당 메세지를 Langchain AmazonKendraReteriver를 활용하여 Kendra에 질문과 관련한 문서가 있는지 검색합니다.
5. 검색한 결과를 기반으로 Prompt Engineering 기반으로 Bedrock을 통해 답변을 생성합니다. 답변이 불가능 할 시 모른다고 답변하도록 설정합니다.
6. Bedrock을 통해 생성한 답변을 REST 기반으로 Slack 채널로 전달합니다.

## 데모 결과
<img src="/Picture2.png"></img>

## 데모 환경
- Region : us-east-1
- Kendra
    - Datasource - webcrawler2
        - URL : https://kakaopaysec.com/
        - Depth : 5
- Bedrock
    - Model :anthropic.claude-v2
    - Token : 2000

## 코드

### Response Lambda
<pre><code>
import json
import boto3
import re
import os
    
def lambda_handler(event, context):

    slackBody = json.loads(event['body'])
    slackText = slackBody.get('event').get('text')
    slackUser = slackBody.get('event').get('user')
    channel =  slackBody.get('event').get('channel')
    
    body = {
        'slackBody' : slackBody,
        'slackText' : slackText,
        'slackUser' : slackUser,
        'channel'   : channel
    }
    
    sql_url = os.environ["sqs_url"]
    sqs_client = boto3.client('sqs')
    
    response = sqs_client.send_message(
        QueueUrl=sql_url,
        MessageBody=json.dumps(body)
    )
    
    print("Message sent to SQS:", response)
    
    return {
        'statusCode': 200,
        'body': "message successfully response"
    } 
</code></pre>

### Langchain Lambda

<pre><code>
import json
import boto3
import urllib3
import re
import requests
import os

from langchain_community.retrievers import AmazonKendraRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.llms.bedrock import Bedrock

bedrock = boto3.client(service_name='bedrock-runtime')
slackUrl = os.environ["slack_url"]
REGION="us-east-1"
KENDRA_INDEX_ID = os.environ["kendra_index"]
BEDROCK = boto3.client(service_name='bedrock-runtime')

def build_chain():
  
  llm = Bedrock(
      model_kwargs={"max_tokens_to_sample":2000,"temperature":1,"top_k":250,"top_p":0.999,"anthropic_version":"bedrock-2023-05-31"},
      model_id="anthropic.claude-v2",
      client=BEDROCK
  )
  
  attributeFilter = {
        "EqualsTo": {
            "Key": "_language_code",
            "Value": {
                "StringValue": "ko"
            }
        },
    }
      
  retriever = AmazonKendraRetriever(
      index_id=KENDRA_INDEX_ID,
      top_k=5,
      region_name=REGION,
      attribute_filter=attributeFilter
      )

  prompt_template = """사람: 이것은 인간과 인공지능 간의 대화입니다. 
  AI는 대화가 가능하며 문맥에서 구체적인 세부 정보를 제공하지만 2000 토큰으로 제한됩니다.
  AI가 질문에 대한 답을 모를 경우, 정직하게 다음과 같이 말합니다. 
  모른다고 말합니다.

  Assistant: 알겠습니다, 진실한 AI 비서가 되겠습니다..

  Human: 다음은 <documents> 태그에 있는 몇 가지 문서입니다.:
  <documents>
  {context}
  </documents>
  {question}에 대한 자세한 답변을 직접 제공할 수 없는 경우 위의 문서를 참조하세요.
  그래도 답변을 제공할 수 없는 경우 '모르겠습니다'라고 답하세요.

  Assistant:
  """
  
  PROMPT = PromptTemplate(
      template=prompt_template, input_variables=["context", "question"]
  )
  qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        return_source_documents=True, 
        combine_docs_chain_kwargs={"prompt":PROMPT},
        verbose=True)

  return qa
 
def run_chain(chain, prompt: str, history=[]):
    result = chain({"question": prompt, "chat_history": history})
    return {
        "answer": result['answer'],
        "source_documents": result['source_documents']
    }
    
    
def publishMessage(url, user, message) :

    data = {'text': f"<@{user}> {message}"}
    
    headers = {
        'Content-Type': 'application/json',
    }
    
    response = requests.post(slackUrl, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        print("Success: Request completed successfully.")
        
    else:
        print(f"Error: Request failed with status code {response.status_code}.")
        raise Exception("Request failed")
    
def lambda_handler(event, context):
    
    data = json.loads(json.dumps(event))
    record = data['Records'][0]
    body = json.loads(record['body'])
    slack_text = body.get('slackText')
    slack_user = body.get('slackUser')
    
    chat_history = []
    chain = build_chain()
    
    result = run_chain(chain, slack_text, chat_history)
    
    answer = ""
    answer += "답변 : \n"
    answer += f"{result['answer']}\n"
    answer += "출처 : \n"

    if 'source_documents' in result:
        for d in result['source_documents']:
          answer += f"{d.metadata['source']}\n"

    print(f"slackText :{slack_text}, slackUser : {slack_user}")
   
    publishMessage(slackUrl, slack_user, answer)
       
    return {
        'statusCode': 200,
    }
</code></pre>

## 주요 이슈 해결과정 및 Lesson Learned

**1. Lambda 호출이 여러번되어 답변이 여러번 발생하는 문제**
- **문제** : Slack을 통해서 질문을 했을때, 아래와 같이 답변이 여러번 오는 문제가 발생함.
<img src="/Picture3.png"></img>

- **원인** : Slack App에서 사용자가 던진 질문을 Lambda로 전달하는데, Slack App에서 3초 안에 답변을 받지 못하면 여러번 재시도 하는 로직이 있었음.
<img src="/Picture4.png"></img>

- **해결 방법** : 아키텍처 상 2번 파이프라인을 만들어 Api gateway를 통해 메세지를 전달받으면 우선 response처리하고, 답변을 생성하는 Lambda를 새로 만들어 SQS로 라우팅하였음.

**2. LangChain Lambda Layer 에러 문제**

- **문제** : Langchain Libray를 Lambda Layer로 등록했는데, 아래와 같이 사용하지도 않는 Library로 인한 문제가 발생했음.

<pre><code>[ERROR] Runtime.ImportModuleError: Unable to import module 'lambda_function': No module named 'orjson.orjson'
Traceback (most recent call last):
</code></pre>
- **원인** : Local Mac에서 Langchain Library를 다운로드 받아 생성하면 생기는 문제로 보임[(관련 링크)](https://stackoverflow.com/questions/75140271/unable-to-import-module-lambda-function-no-module-named-orjson-orjson)
- **해결 방법** : Python Docker를 올려서 Docker 안에서 Library 다운로드 및 python.zip을 만드니 해결되었음.([Public ECR URL](https://gallery.ecr.aws/))

<pre><code>sudo docker pull public.ecr.aws/sam/build-python3.11:1.110.0-20240222205823
docker run -it -v /Users/jinsungh/Desktop/rag:/var/task public.ecr.aws/sam/build-python3.11:1.110.0-20240222205823
pip install langchain -t ./python
zip -r python.zip ./python
</code></pre>

**3. Kendra에서 관련 문서를 찾지 못하는 문제**

- **문제** : 아래와 같이 KendraRetreiver를 사용하는데 질문과 관련된 문서를 찾지 못하는 문제가 발생함.
<pre><code>Unfortunately I do not have access to any documents in my context, so I do not know the detailed steps for opening a securities account with KakaoPay through mobile. Since I do not have the relevant information, the honest response is "I don't know". I apologize that I cannot provide a more helpful answer without additional context
</code></pre>
- **원인** : KendraRetriever를 사용시 Language 설정이 필요했음. 그런데 이 내용은 찾아도 안나오던 사항
- **해결 방법** : 아래와 같이 KendraRetriever를 생성 시 ‘attribute_filter’ parameter 값을 넣어서 language 속성을 추가 함.

<pre><code>attributeFilter = {
        "EqualsTo": {
            "Key": "_language_code",
            "Value": {
                "StringValue": "ko"
            }
        },
    }
      
  retriever = AmazonKendraRetriever(
      index_id=KENDRA_INDEX_ID,
      top_k=5,
      region_name=REGION,
      attribute_filter=attributeFilter
      )
</code></pre>