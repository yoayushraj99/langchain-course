import os

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()  # Load environment variables from .env file


def main():
    print("Hello from langchain-course!")

    information = """Ryan Thomas Gosling (/ˈɡɒslɪŋ/ GOSS-ling;[1] born November 12, 1980) is a Canadian actor. His work includes both independent films and major studio features, and his accolades include a Golden Globe Award, in addition to nominations for three Academy Awards, a Primetime Emmy Award, and two British Academy Film Awards.

Gosling began his acting career when he was 13 on Disney Channel's The All New Mickey Mouse Club (1993–1995), and went on to appear in other family entertainment programs, including Are You Afraid of the Dark? (1995) and Goosebumps (1996). His film breakthrough came as a high school football player in Remember the Titans (2000) followed by critical praise for his performance as a Jewish neo-Nazi in The Believer (2001). In 2004, he achieved mainstream popularity for starring in the romantic drama The Notebook.

Gosling earned Academy Award nominations for his performances in the independent drama Half Nelson (2006), the romantic musical La La Land (2016), and the fantasy comedy Barbie (2023); the latter became his highest-grossing release. He also starred in acclaimed films such as Lars and the Real Girl (2007), Blue Valentine (2010), Crazy, Stupid, Love (2011), The Ides of March (2011), Drive (2011), The Big Short (2015), Blade Runner 2049 (2017), First Man (2018), and Project Hail Mary (2026). Gosling made his directorial debut with the film Lost River in 2014.

Gosling's band, Dead Man's Bones, released their self-titled debut album and toured North America in 2009. He is a co-owner of Tagine, a Moroccan restaurant in Beverly Hills, California. Gosling is a supporter of People for the Ethical Treatment of Animals (PETA), Invisible Children, and the Enough Project and has traveled to Chad, Uganda and eastern Congo to raise awareness about conflicts in the regions. He has been involved in peace promotion efforts in Africa for over a decade.

"""

    summary_template = """
    Given the information {information} about a person I want you to create:
    1. A short summary of the person
    2. Two interesting facts about the person
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()