from dotenv import load_dotenv

load_dotenv() 

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo")

from crewai import Agent
from crewai import Task
from crewai import Crew, Process


class CoverLetterAgents:
    def cover_letter_expert(self):
        return Agent(
            role="Cover Letter Expert",
            goal="Generate a compelling cover letter for visa application",
            backstory="Experienced in crafting persuasive cover letters for visa applications",
            verbose=True,
            llm=llm,
        )

    def cover_letter_proofread(self):
        return Agent(
            role="Cover Letter Proofreader",
            goal="Proofread and refine the cover letter for accuracy and clarity",
            backstory="Skilled in editing and polishing cover letters to meet high standards",
            verbose=True,
            llm=llm,
        )


class CoverLetterTasks:
    def generate_cover_letter_task(self, agent, from_country, to_country, occupation):
        return Task(
            description=f"Generate a compelling cover letter for the visa application from {from_country} to {to_country}, emphasizing the applicant's occupation as a {occupation}.",
            agent=agent,
            expected_output="String Format",
        )

    def generate_cover_letter_proofread_task(self, agent):
        return Task(
            description="Proofread and refine the cover letter for accuracy and clarity.",
            agent=agent,
            expected_output="A refined version of cover letter in string format",
        )


class CoverLetterCrew:
    def __init__(self, from_country, to_country, occupation):
        self.from_country = from_country
        self.to_country = to_country
        self.occupation = occupation

    def run(self):
        agents = CoverLetterAgents()
        tasks = CoverLetterTasks()

        cover_letter_expert = agents.cover_letter_expert()
        cover_letter_proofread = agents.cover_letter_proofread()

        generate_cover_letter_task = tasks.generate_cover_letter_task(
            cover_letter_expert, self.from_country, self.to_country, self.occupation
        )

        generate_cover_letter_proofread_task = (
            tasks.generate_cover_letter_proofread_task(cover_letter_proofread)
        )

        crew = Crew(
            agents=[cover_letter_expert, cover_letter_proofread],
            tasks=[generate_cover_letter_task, generate_cover_letter_proofread_task],
            verbose=True,
        )

        result = crew.kickoff()
        return result


if __name__ == "__main__":
    trip_crew = CoverLetterCrew("India", "France", "Writer")
    result = trip_crew.run()
    print("\n\n########################")
    print("## Here is your Cover Letter")
    print("########################\n")
    print(result)
