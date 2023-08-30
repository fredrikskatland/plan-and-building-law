import streamlit as st

from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import MessagesPlaceholder
from langsmith import Client
from langchain.vectorstores import Chroma

import os


local = False


client = Client()

st.set_page_config(
    page_title="Plan and Building Law",
    page_icon=":scales:",
    layout="wide",
    initial_sidebar_state="collapsed",
)



original_system_message = SystemMessage(
    content=(
        "You are a helpful chatbot who is tasked with answering questions about the contents of the Plan and Building Law. "
        "Unless otherwise explicitly stated, it is probably fair to assume that questions are about the Plan and Building Law. "
        "If there is any ambiguity, you probably assume they are about that."
    )
)

summary_system_message = SystemMessage(
    content=(
       """You are a helpful chatbot who is tasked with answering questions about the contents of the Plan and Building Law. 
            Unless otherwise explicitly stated, it is probably fair to assume that questions are about the Plan and Building Law. 
            If there is any ambiguity, you probably assume they are about that.
            This is a summary of the Plan and Building Law:
                
Part I: General provisions
Chapter 1. Common provisions
Section 1-1. Purpose of the Act
- The Act aims to promote sustainable development for the benefit of individuals, society, and future generations.
- Planning under this Act should facilitate coordination between central government, regional, and municipal functions and serve as a basis for administrative decisions on resource use and conservation.
- Building applications should ensure compliance with laws, regulations, and planning decisions, and projects should be carried out properly.
- Planning and administrative decisions should prioritize transparency, predictability, and public participation, with a focus on long-term solutions and describing environmental and social impacts.
- Design for universal accessibility and consideration for the environment in which children and youth grow up should be taken into account in planning and building requirements.

Section 1-2. Scope of the Act
- The Act applies to the entire country, including river systems.
- In marine areas, the Act applies to a zone extending one nautical mile beyond the territorial sea baselines.
- The King may decide to apply Chapter 14 to specific projects beyond one nautical mile.
- The Act may also apply, in whole or in part, to Svalbard.

Section 1-3. Projects that are exempted from the Act
- The Act does not apply to marine pipelines for petroleum transport.
- Only chapters 2 and 14 apply to installations for electric power transmission or conversion mentioned in the Energy Act.
 
Section 1-4. The functions of planning and building authorities and the duties of other authorities
- Planning and building authorities are responsible for ensuring compliance with planning and building legislation in municipalities.
- They should collaborate with other public authorities involved in matters under the Planning and Building Act and seek their opinions.
- If any person conducting a public inspection finds non-compliance with the Act, they should report it to the planning and building authorities.

Section 1-5. Effects of plans
- The impact of plans adopted under this Act on further planning, management, and administrative decisions is determined by the provisions for different types of plans.
- In case of conflict, a new plan or central government/regional planning provision supersedes an older plan or provision, unless otherwise specified.

Section 1-6. Projects
- A "project" under this Act includes building erection, demolition, alteration (including exterior alteration), changes in use, land alteration, property establishment/alteration, and activities/alterations contrary to land-use objectives, planning provisions, and special consideration zones.
- Projects within the Act's scope can only proceed if they comply with the Act, regulations, municipal master plan's land-use element, and zoning plan.
- Certain projects may be exempt from application and permits or may be carried out by the developer themselves.

Section 1-7. Joint processing of planning and building applications
- General permission applications can be submitted along with zoning plan proposals and processed together.

Section 1-8. Prohibition on projects along the seashore and river systems
- The 100-meter belt along the seashore and river systems should consider natural/cultural environment, outdoor recreation, landscape, and public interest.
- Projects, except exterior building alterations, cannot be implemented closer to the sea than 100 meters from the shoreline, unless specified otherwise in the municipal master plan's land-use element or zoning plan.
- Exceptions include sectioning off land for developed leasehold site redemption and permitted necessary buildings/installations/storage facilities for agriculture, reindeer husbandry, fishing, aquaculture, or sea traffic.
- For significant areas along river systems, the municipal master plan can set a limit of up to 100 meters where specific projects are not permitted.

Section 1-9. Relationship to the Public Administration Act and appeals
- The Public Administration Act applies with special provisions from this Act.
- No appeal can be lodged regarding matters decided in a binding zoning plan or by dispensation, where the time limit for appealing has expired. Appeals may be considered if deemed expedient.
- Central government, regional, or municipal bodies affected by individual decisions under this Act can appeal if the decision directly affects their responsibility, except in planning matters where objections are allowed.
- Access to documents is granted under the Freedom of Information Act and the Environmental Information Act, with exceptions from the Freedom of Information Act's chapter 3.
- The Ministry serves as the administrative appeals body for individual decisions under this Act.

Chapter 2. Requirements relating to basic map data, geodata, etc.
Section 2-1. Maps and geodata
- Municipalities must have an up-to-date, public set of basic map data for specified objectives.
- National map data should be made available to all municipalities by central government authorities.
- Geodata should be organized to ensure easy access for processing planning and building applications, and should also be available for other public and private purposes.
- Municipalities may require plan proposals and project applications to include necessary maps, which can be incorporated into public sector basic map data. Digital submission may be required.
- The King may initiate projects to collect, check, revise, or supplement planning/building information and basic map data, and public bodies may be ordered to provide necessary information.
- The Ministry can make regulations regarding maps and geodata.

Section 2-2. Municipal planning registers
- Municipalities must have a planning register providing information on current land-use plans and other provisions determining land use.
- The Ministry can make regulations regarding municipal planning registers, including electronic registers.

Part II: The planning part
The Planning and Building Act consists of several chapters that outline the provisions and responsibilities related to planning functions and authority. 

Chapter 3 focuses on planning functions and considerations. Plans under this Act are required to establish goals for the development of municipalities and regions, identify social needs and functions, and state how these functions can be discharged. They must also safeguard land resources, landscape qualities, and valuable landscapes and cultural environments. Additionally, plans should protect the natural basis for Sami culture, facilitate value creation and industrial development, promote good design and living standards, promote public health and safety, and take climate into account. Planning should promote coherence and collaboration between sector authorities and various bodies involved in planning. Plans should be based on financial and resource-related prerequisites and contribute to the implementation of international conventions. Adopted plans serve as a basis for planning activities by municipal, regional, central government, and private-sector entities.

Section 3-2 discusses the responsibility for and assistance in planning. Municipal councils, regional planning authorities, and the King are responsible for planning under this Act. Public and private bodies have the right to propose planning projects, and all public bodies are required to participate in planning within their sphere of responsibility. The county governor ensures that municipalities fulfill their planning duty, and regional planning authorities guide and assist municipalities in their planning functions.

Section 3-3 focuses on municipal planning functions and authority. The purpose of municipal planning is to provide favorable conditions for development and coordinated functions within the municipality. The municipal council directs the planning process and ensures compliance with planning and building legislation. The council adopts a municipal planning strategy, a master plan, and a zoning plan. The municipality organizes the planning work and establishes committees as necessary. The council also ensures the interests of children and youth are safeguarded in the planning process and has access to necessary planning expertise.

Section 3-4 discusses regional planning functions and authority. The purpose of regional planning is to stimulate development in various aspects. The regional planning authority is responsible for formulating a regional planning strategy, regional master plans, and regional planning provisions. The authority cannot delegate its competence to adopt a regional master plan. The authority ensures the availability of planning expertise and is typically the county council.

Section 3-5 addresses central government planning functions and authority. Central government planning guidelines and decisions aim to safeguard national or regional interests in planning. The King is responsible for directing planning activities at the national level, and the Ministry has administrative responsibility for central government planning functions. The Ministry oversees compliance with regional planning duties.

Section 3-6 discusses common planning functions. The central government and regional authorities can collaborate on plans of regional or national significance. This includes coordinated land-use and transport planning, planning of nature and outdoor recreation areas, coordinated water planning, and coastal zone planning. The King makes regulations specifying the functions, areas, and authorities involved in common planning provisions.

Section 3-7 addresses the transfer of planning preparations to the central government or regional authority. By agreement, the central government or regional authority can take over planning functions from the municipal planning administration and regional planning authorities. If no agreement is reached, the Ministry makes the decision. Authorities responsible for major infrastructure projects can prepare and present land-use plans for such projects after consultation with planning authorities.

Chapter 4 focuses on general requirements regarding assessments. Section 4-1 discusses the planning program, which is required for regional and municipal plans and zoning plans with significant impacts. The program provides a basis for the planning work and includes the purpose of the planning, the planning process, arrangements for public participation, alternatives considered, and the need for assessments. The program is determined by the planning authority. If the plan may conflict with national or regional considerations, affected authorities can state their concerns. If the plan has significant environmental impacts in another state, the proposal is sent for comment to the authorities in that state.

Section 4-2 discusses the description of the plan and impact assessment. All proposals for plans must include a plan description specifying objectives, contents, effects, and the impact of limits and guidelines. For plans with substantial environmental and societal effects, a separate impact assessment is required. The King can make regulations regarding planning programs, descriptions, and impact assessments.

Section 4-3 focuses on societal safety and risk and vulnerability assessments. Planning authorities must ensure that a risk and vulnerability assessment is carried out for the planning area. The assessment identifies factors significant for determining the suitability of land for development and any changes resulting from planned development. Areas with danger, risk, or vulnerability must be indicated in the plan, and provisions may be adopted to prevent damage and loss. The King can make regulations regarding risk and vulnerability assessments.

Chapter 5 addresses public participation in planning. Section 5-1 states that anyone presenting a planning proposal must facilitate public participation. The municipality is responsible for ensuring participation, especially for groups requiring special facilitation. Section 5-2 discusses consultation and public scrutiny, requiring proposals to be sent for comment to affected authorities and made accessible to the public. Electronic presentation and dialogue should be facilitated. Section 5-3 mentions the establishment of regional planning forums to coordinate interests in regional and municipal plans. Section 5-4 allows affected bodies to make objections to planning proposals, and Section 5-5 outlines limitations on the right to make objections. Section 5-6 discusses mediation and decision by the Ministry in case of disagreement between the municipality and objecting parties.

The Planning and Building Act includes several sections related to national planning functions. 

Section 6-1 states that every four years, the King will create a document outlining national expectations for regional and municipal planning in order to promote sustainable development. This document will serve as the basis for central government participation in planning activities.

Section 6-2 allows the King to issue central government planning guidelines for the entire country or specific geographic areas. These guidelines will serve as the basis for central government, regional, and municipal planning, as well as individual decisions made by these bodies.

Proposals for central government planning guidelines must be circulated for comment for a period of six weeks before being adopted. The Ministry is responsible for making these guidelines known to all affected public bodies, interested organizations, institutions, and the general public.

Section 6-3 grants the King the authority to impose a prohibition on certain building or installation projects without the consent of the Ministry. This prohibition can last for a maximum of ten years and can be extended for five-year periods. Before making a decision, proposals for provisions must be circulated for comment and presented for public scrutiny within the affected municipalities. Once a decision is made, the central government planning provisions will be announced in the Norwegian Legal Gazette and made known to all affected parties.

Section 6-4 allows the Ministry to request a municipality to prepare the land-use element of the municipal master plan or the zoning plan for important central government or regional projects. The Ministry may also prepare and adopt such a plan themselves, assuming the authority of the municipal council. The municipality must provide necessary assistance in this process. Additionally, the Ministry has the power to decide that a final license for a power plant will automatically have the effect of a central government land-use plan, and these decisions cannot be appealed.

III. Regional planning
The Planning and Building Act, specifically in Chapter 7, Section 7-1, states that the regional planning authority must prepare a regional planning strategy in collaboration with municipalities, central government bodies, organizations, and institutions affected by the planning work. This strategy should address important regional development trends and challenges, assess long-term development potentials, and determine which issues should be addressed through further regional planning. The strategy should also include an overview of how prioritized planning functions will be followed up and the arrangements for public participation in planning work. The King has the authority to make regulations regarding the contents and arrangements of the regional planning process.

In Section 7-2, it is stated that proposals for regional planning strategies must be circulated for comment and presented for public scrutiny. The time limit for submitting comments should be at least six weeks. Once adopted by the regional planning authority, the strategy is submitted to the King for approval. The King may make alterations to the strategy for reasons related to national interests. Central government bodies, regional bodies, and municipalities are required to use the regional planning strategy as the basis for further planning work in the region.

Moving on to Chapter 8, Section 8-1, it is stated that the regional planning authority must prepare regional master plans that address the issues identified in the regional planning strategy. The King has the authority to make orders regarding the preparation of regional master plans for specific areas of activity, topics, or geographical areas. The regional master plan should be implemented through a program of action, which is adopted by the regional planning authority and rolled over annually.

Section 8-2 states that the regional master plan serves as the basis for the activities of regional bodies, as well as municipal and central government planning and activities in the region.

Section 8-3 outlines the process for preparing regional master plans. The regional planning authority must collaborate with affected public authorities and organizations. Central government bodies and municipalities have the right and duty to participate in the planning process if it affects their area of activity or their own plans and decisions. A proposal for a planning program should be prepared in cooperation with affected municipalities and central government authorities. The proposal should be circulated for comment and presented for public scrutiny. Proposals for regional master plans should also be circulated for comment and presented for public scrutiny. These plans should include a special assessment and description of the environmental and societal effects.

Section 8-4 explains the adoption of regional master plans. The regional master plan is adopted by the regional planning authority, unless the matter is submitted to the Ministry or other provisions are made by regulations. If a central government body or municipality has substantial objections to the goals or guidelines of the plan, they may demand that the matter be submitted to the Ministry for alterations. The Ministry may also make alterations to the plan based on national interests. The regional planning authority must be informed of any alterations made by the Ministry.

Section 8-5 discusses regional planning provisions related to guidelines for land use in a regional master plan. These provisions may impose a prohibition on certain building or construction projects without consent within delimited geographical areas. Proposals for regional planning provisions should be prepared and considered in accordance with the provisions of Sections 8-3 and 8-4. The regional planning authority may extend the prohibition for five years at a time. Consent for projects to which a regional planning provision applies may be given by the regional planning authority after consultation with the county governor and affected municipalities. Regional planning provisions should be announced in the Norwegian Legal Gazette and made available through electronic media.

Moving on to Chapter 9, Section 9-1 states that two or more municipalities should cooperate on planning when it is necessary to coordinate planning across municipal borders. Intermunicipal planning cooperation may be initiated as the implementation of a regional planning strategy. The regional planning authority or central government authorities may request municipalities to enter into such cooperation when necessary to safeguard considerations and discharge functions beyond the individual municipality. The Ministry may also order municipalities to enter into planning cooperation when necessary to safeguard national and important regional considerations and functions.

Section 9-2 discusses the organization of intermunicipal planning cooperation. The planning work is directed by a board consisting of an equal number of representatives from each municipality, unless otherwise agreed upon. The board establishes rules for its work and organizes the planning process.

Section 9-3 states that the provisions regarding the type of planning and content of plans apply to intermunicipal planning cooperation. Each municipality is responsible for ensuring compliance with the rules of procedure in its area. The participating municipalities may delegate decision-making authority to the board. Final planning decisions for each municipality are made by their respective municipal council.

Section 9-4 allows a majority of municipalities to request the regional planning authority to take over the planning work as a regional master plan. The regional planning authority and central government authority may also request municipalities to continue the work as a regional master plan. The Ministry may decide to continue the planning work as a regional master plan and must give the municipalities an opportunity to express their views.

Section 9-5 addresses disagreements in joint planning proposals. Mediation can be requested if municipalities disagree on the content of a joint planning proposal. A municipality may withdraw from cooperation by giving three months' written notice, while the other municipalities may continue their planning cooperation. The Ministry may order individual municipalities to continue participating in cooperation.

Section 9-6 discusses the implementation and alterations of plans in intermunicipal planning cooperation. Agreements may be made regarding the implementation of plans adopted through cooperation. If a municipality or regional planning authority wishes to unilaterally alter a plan, written notice must be given to the other participants and affected parties.

Section 9-7 states that the provisions of Chapter 9 also apply to planning cooperation between regions and municipalities. The Ministry may impose planning cooperation when necessary to discharge planning functions for large areas collectively. The regions and municipalities involved must be given an opportunity to express their views before such provisions are made.


IV. Municipal planning
The Planning and Building Act includes several sections related to municipal planning and the municipal master plan. 

Chapter 10 focuses on the municipal planning strategy. Section 10-1 states that the municipal council must prepare and adopt a municipal planning strategy at least once in each electoral term. This strategy should discuss the municipality's strategic choices related to social development, long-term land use, environmental challenges, sector activities, and planning needs. The municipality should seek the views of central government, regional bodies, neighboring municipalities, and promote public participation in the development of the strategy. Proposals for council decisions must be made public at least 30 days before they are considered.

Chapter 11 discusses the municipal master plan. Section 11-1 states that municipalities must have an overall municipal master plan that includes a social element, an implementation element, and a land-use element. The plan should promote municipal, regional, and national goals and cover all important goals and functions in the municipality. It should be based on the municipal planning strategy and guidelines from central government and regional authorities. The plan may also include specific sub-plans for certain areas or topics.

Section 11-2 focuses on the social element of the municipal master plan. This element should determine long-term challenges, goals, and strategies for the municipality as a whole and as an organization. It should serve as a basis for sector plans and activities and provide guidelines for implementing the municipality's goals and strategies. Municipal sub-plans should include an implementation element that is revised annually.

Section 11-3 states that the social element of the municipal master plan should serve as the basis for the municipality's own activities, as well as the activities of central government and regional authorities in the municipality. The implementation element of the plan should guide resource prioritization, planning, and cooperation functions within the municipality's financial framework.

Section 11-4 discusses the revision of the social element of the municipal master plan and the municipal sub-plan. The provisions regarding the municipal planning strategy and consideration of the municipal master plan apply to the revision of the social element and sub-plans. The implementation element of the municipal master plan should be reviewed annually, and proposals for decisions should be made public at least 30 days before they are considered by the municipal council.

Section 11-5 focuses on the land-use element of the municipal master plan. Municipalities should have a land-use plan for the entire municipality that shows the connection between future social development and land use. The plan should state the allocation of land, frameworks, conditions for new projects, and important considerations for land allocation. The plan should include a planning map, provisions, and a description of how national goals and guidelines have been complied with. The municipality may provide a detailed description of the land-use element for specific areas, including sub-objectives and provisions.

Section 11-6 states that the land-use element of the municipal master plan determines future land use and is binding for new projects or expansions. Projects must comply with the plan's objectives and provisions. The plan should be followed when deciding permits or managing projects.

Section 11-7 outlines the land-use objectives in the land-use element of the municipal master plan. These objectives include buildings and installations, transport and communication installations, green structures, the Norwegian Armed Forces, agricultural and nature objectives, and the use and conservation of sea and river systems. The plan may include sub-objectives for each objective.

Section 11-8 discusses zones requiring special consideration in the land-use element of the municipal master plan. These zones may include safety, noise, and danger zones, zones requiring special infrastructure, zones with considerations for agriculture, reindeer husbandry, outdoor recreation, green structures, landscape, or protection of the natural or cultural environment. The plan should include provisions and guidelines for each zone.

Section 11-9 provides general provisions for the land-use element of the municipal master plan. These provisions include requirements for zoning plans, content of development agreements, requirements for specific solutions for water supply, sewerage, roads, and other transport, requirements for the order of implementing public services and infrastructure, building limits, environmental quality, conservation of existing buildings and cultural elements, and further zoning work.

Chapter 12 focuses on zoning plans. Section 12-1 states that zoning plans are land-use plan maps with provisions for land use, conservation, and design. The municipal council is responsible for preparing zoning plans for areas where it is required by the Act or the municipal master plan. Zoning plans are necessary for major building projects and projects with significant environmental effects. Permits for such projects cannot be granted until a zoning plan exists.

Section 12-2 discusses area zoning plans, which are used to clarify land use in greater detail area by area. These plans are prepared by the municipality or other authorities and private bodies.

Section 12-3 focuses on detailed zoning plans, which follow the land-use element of the municipal master plan and area zoning plans. Private bodies, developers, and other authorities can propose detailed zoning plans for specific projects. These plans must comply with the main features and limits of the land-use element and existing area zoning plans.

Section 12-4 states that a zoning plan is binding for new projects or extensions once adopted by the municipal council. Projects must comply with the plan's objectives and provisions. The plan is also a basis for expropriation.

The Act also includes sections on the temporary prohibition of projects, the duration and time limits for such prohibitions, and the central government's authority to prohibit building and sectioning in certain cases.

V. Impact assessments in relation to projects and plans pursuant to other legislation
Chapter 14 of the Planning and Building Act focuses on impact assessments for projects and plans that may have significant impacts on the environment and society. This chapter also applies to specific conservation plans under the Nature Conservation Act. The purpose of these provisions is to ensure that the environment and society are considered during the preparation of the project or plan, and when deciding whether or not to implement it.

Section 14-1 states that these provisions apply to projects and plans under other legislation that may have substantial impacts on the environment and society. It also applies to specific conservation plans under the Nature Conservation Act. The purpose is to consider the environment and society during the preparation of the project or plan, and when deciding whether or not to implement it.

Section 14-2 outlines the process for preparing and considering assessment programs and impact assessments for projects and plans covered by these provisions. A notice with a proposed assessment program should be prepared early in the project or plan's preparation. This proposal should include information about the project, the need for assessments, and arrangements for public participation. The proposed program should be circulated for comment and presented for public scrutiny before it is finalized. An application or planning proposal with an impact assessment should be prepared based on the finalized assessment program and circulated for comment and presented for public scrutiny.

Section 14-3 states that the decision on the project or plan should consider the impact assessment and any comments received. The decision should show how the impacts were assessed and what significance they have in the decision-making process, particularly in regards to alternative choices. The decision and its reasons should be made public. Conditions may be imposed to monitor and mitigate any significant negative impacts, and these conditions should be included in the decision.

Section 14-4 addresses the need to consider transboundary effects of projects or plans covered by this chapter. If a project or plan may have significant negative environmental impacts in another state, the responsible authority should inform the affected authorities in that state and give them an opportunity to participate in the planning or assessment process.

Section 14-5 states that the costs of preparing a notice with a proposal for an assessment program and impact assessment should be borne by the proposer.

Section 14-6 gives the King the authority to make regulations regarding which projects and plans are covered by this chapter and to provide supplementary provisions for assessment programs and impact assessments.

Chapter 15 of the Act focuses on redemption and compensation for landowners. Section 15-1 states that if an undeveloped property is designated in the municipal master plan for certain purposes and is not zoned or designated for other purposes within four years, the landowner or lessee may demand compensation or immediate expropriation if the property can no longer be utilized profitably. The same rights apply if the property is developed and the buildings are removed.

Section 15-2 states that if a zoning plan allows for expropriation of an undeveloped property or part of a property, the landowner or lessee may demand immediate expropriation if the decision relates to certain designated purposes. The same rights apply if the expropriation makes the property no longer profitable. Claims for compensation must be filed within three years of the zoning plan being announced or the decision being made known.

Section 15-3 addresses compensation for loss in connection with a zoning plan. If a zoning plan spoils a property as a building site or makes it no longer profitable for agricultural purposes, the municipality must pay compensation based on appraisement unless they acquire the property. Claims for compensation must be filed within three years of the zoning plan being announced or the decision being made known.

Chapter 19 of the Act focuses on dispensation. Section 19-1 states that a reasoned application is required for dispensation. Neighbors must be notified, unless the application does not affect their interests or is filed at the same time as a permit application. Regional and central government authorities affected by the application must have an opportunity to express their views.

Section 19-2 states that the municipality may grant permanent or temporary dispensation from provisions in the Act. Conditions may be imposed for such dispensation. Dispensation may not be granted if the considerations behind the provision are significantly disregarded, and the advantages must outweigh the disadvantages. Dispensation from rules of procedure may not be granted.

Section 19-3 addresses temporary dispensation, which may be granted on a time-limited or indefinite basis. The applicant must remove or alter the work done or cease the temporarily permitted use upon the expiry of the dispensation period or when ordered to do so. Dispensation may be conditional and binding on mortgagees and other rights holders.

Section 19-4 states that the authority to grant dispensation rests with the municipality. In certain cases, the King may assign the authority to a regional or central government body to safeguard national or important regional interests.

The Planning and Building Act consists of several parts, including the final provisions outlined in the sixth part. Chapter 34 of the Act focuses on commencement and transitional provisions. 

Section 34-1 states that the Act will come into force on a date determined by the King. Additionally, the previous Planning and Building Act of 1985 will be repealed from the same date.

Section 34-2 outlines the transitional provisions for the planning part of the Act. Within two years of the Act's commencement, the King must present a document that sets out national expectations for regional and municipal planning. Municipalities and county authorities must also prepare and adopt municipal and regional planning strategies within the first year after the election of a new council.

Existing national policy guidelines and provisions from the previous Planning and Building Act will continue to apply, but any changes to these guidelines and provisions must be made in accordance with the provisions of the new Act. Existing county and municipal master plans, zoning plans, and building development plans will remain in effect until they are amended, revoked, replaced, or set aside by a new plan under the new Act.

Certain limitations on the right of appeal and objections only apply to planning decisions made under the new Act. Provisions related to expropriation, site preparation, reimbursement, processing of applications, and sanctions under the previous Act will continue to apply to plans prepared before the new Act's commencement.

Earlier zoning plans and building plans will still serve as a basis for expropriation within a ten-year time limit. An exception related to buildings, structures, installations, or fencing necessary in agriculture will remain in effect until new provisions are adopted, but will cease to apply four years after the Act's commencement.

Municipal regulations and bylaws will continue to apply until they are replaced by new planning provisions, regulations, or bylaws. However, certain municipal bylaws will cease to apply no later than eight years after the Act's commencement, unless dispensation is granted.

Proposals for the land-use element of a municipal master plan, zoning plan, or development plan that were presented for public scrutiny upon the Act's commencement may be finalized under the provisions in effect at that time. Other plans will be subject to the provisions of the new Act.

For projects requiring an impact assessment under the provisions of the previous Act's chapter VII-a, and where the planning program has been approved, the impact assessment may be completed under those provisions.

The Ministry has the authority to make further regulations concerning the functioning of the previous Planning and Building Act in conjunction with the new Act.

The contact information for the Housing and Building Department and the Department for Planning is provided, including email addresses, phone numbers, and addresses.

"""
    )
)   

@st.cache_resource(ttl="1h")
def configure_retriever():
    if not os.path.exists('./PlanAndBuilding_chroma_db'):
        #print("does not exists")
        loader = DirectoryLoader("./sections")
        docs = loader.load()
        if local:
            embeddings = OpenAIEmbeddings()
        else:
            embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_api_key"])

        docsearch = Chroma.from_documents(docs, embeddings, persist_directory="./PlanAndBuilding_chroma_db")
        print("Persisting to disk: PlanAndBuilding_chroma_db")
        docsearch.persist()

        retriever = docsearch.as_retriever()

        return retriever
    else:
        if local:
            embeddings = OpenAIEmbeddings()
        else:
            embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_api_key"])
        # load from disk
        print("loading from disk: PlanAndBuilding_chroma_db")
        docsearch = Chroma(persist_directory="./PlanAndBuilding_chroma_db", embedding_function=embeddings)
        retriever = docsearch.as_retriever()
        return retriever

def reload_llm(model_choice="gpt-4", temperature=0, system_message_choice="Original"):
    if local:
        llm = ChatOpenAI(temperature=temperature, streaming=True, model=model_choice, )
    else:
        llm = ChatOpenAI(temperature=temperature, streaming=True, model=model_choice, openai_api_key=st.secrets["openai_api_key"])

    if system_message_choice == "Original":
        message = original_system_message
    elif system_message_choice == "New (extensive summary)":
        message = summary_system_message

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
    )

    tool = create_retriever_tool(
        configure_retriever(),
        "search_plan_and_building_law",
        "Search Plan and Building Law. This tool should be used when you want to get information from the Plan and Building Law."
    )
    tools = [tool]

    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
    )
    memory = AgentTokenBufferMemory(llm=llm)
    print ("Reloaded LLM")
    return agent_executor, memory, llm


# Using "with" notation
with st.sidebar:
    with st.form('my_form'):
        model_choice = st.radio(
            "Model",
            ("gpt-4", "gpt-3.5-turbo-16k")
        )
        temperature = st.slider('Temperature', 0.0, 1.0, 0.0, 0.01)
        system_message_choice = st.radio(
            "System message",
            ("Original", "New (extensive summary)")
        )
        submitted = st.form_submit_button('Reload LLM')
    if submitted: 
        reload_llm(model_choice=model_choice, temperature=temperature)
        print(model_choice, temperature)
    

"# Chat with the Plan and Building Law :scales: 🔗"


starter_message = "Skrivebok for Bendik Svartva.."
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [AIMessage(content=starter_message)]

agent_executor, memory, llm = reload_llm()

for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    memory.chat_memory.add_message(msg)


if prompt := st.chat_input(placeholder=starter_message):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        agent_executor, memory, llm = reload_llm(model_choice=model_choice, temperature=temperature, system_message_choice=system_message_choice)
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor(
            {"input": prompt, "history": st.session_state.messages},
            callbacks=[st_callback],
            include_run_info=True,
        )
        st.session_state.messages.append(AIMessage(content=response["output"]))
        st.write(response["output"])
        memory.save_context({"input": prompt}, response)
        st.session_state["messages"] = memory.buffer
        run_id = response["__run"].run_id
        print(llm)
