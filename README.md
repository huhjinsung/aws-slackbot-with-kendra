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
