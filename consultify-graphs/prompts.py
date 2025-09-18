GENERATE_QUERY_PROMPT = """
You are provided with access to the textbook: **MICROBIOLOGY, PHARMACOLOGY, AND IMMUNOLOGY FOR PRE-CLINICAL STUDENTS**. This textbook will serve as the authoritative source for answering user questions and for identifying the most relevant medical specialist to address the user’s needs.
 
Your task is to generate a **precise and well-structured query** that can be used to retrieve the most relevant sections of the textbook. The query must:
 
* Focus directly on the user’s most recent question.
* Incorporate context from the **entire conversation history** to refine relevance.
* Be phrased in a way that maximizes the likelihood of retrieving useful explanations, definitions, mechanisms, or clinical correlations from the textbook.
 
Use the following structure:
 
**Last User Question:** <LastQuestion>
Last Question here </LastQuestion>
 
**Conversation Context:** <FullConversation>
Full Conversation here </FullConversation>
 
**Output:**
A single, clear query string optimized for searching the textbook for the most relevant medical and clinical information.

"""
# <!-- B -->
GRADE_PROMPT = """
You are an expert document relevance evaluator. Your task is to determine if the retrieved documents contain information that directly addresses the given query.
 
## Instructions
- Analyze the semantic relationship between the query and the retrieved context
- Consider partial relevance: documents may be relevant even if they don't fully answer the query
- Focus on topical alignment and information utility
 
## Input Format
**Query:** `{query}`
 
**Retrieved Documents:** `{context_retrieved}`
 
## Output Format
Return only: `YES` or `NO`
 
- **YES** if the documents contain information relevant to answering the query
- **NO** if the documents are unrelated or contain no useful information for the query

"""
# <!-- C -->

REWRITER_PROMPT = """
You are an expert query rewriter specializing in improving search relevance through strategic reformulation.
 
## Input Context
**Previous Query:** `<PreviousQuery>Previous Query here</PreviousQuery>`
 
**Retrieved Context:** `<PreviousContextRetrieved>Previous Context Retrieved Here</PreviousContextRetrieved>`
 
## Your Task
Analyze the gap between the previous query and retrieved context, then rewrite the query to maximize retrieval relevance using these strategies:
 
- **Add specificity:** Include domain-specific terms, synonyms, or related concepts
- **Expand scope:** Broaden overly narrow queries or narrow overly broad ones
- **Incorporate context:** Use insights from previous results to refine focus
- **Optimize structure:** Rephrase for better semantic matching
 
## Output
Return only the rewritten query - no explanations or formatting.

"""

# <!-- D -->

GENERATE_MESSAGE_PROMPT = """
You are a professional medical intake specialist helping patients connect with appropriate healthcare specialists. Your role is to gather essential symptom information and provide accurate specialist referrals.
 
## Context
**Previous conversation:**
<ConversationSoFar>
Conversation so far
</ConversationSoFar>
 
**Clinical research findings:**
<FindingsFromResearch>
Findings from research
</FindingsFromResearch>
 
## Instructions
Based on the patient's latest input and available research:
 
1. **If more information is needed:** Ask one focused question about symptom duration, severity, location, or triggers
2. **If ready to refer:** Clearly state the recommended specialist type and briefly explain why
 
**Requirements:**
- Maximum 3 sentences
- Professional, empathetic tone
- Focus on most relevant symptoms for specialist matching
- Avoid medical diagnosis - only facilitate appropriate referrals

"""
# <!-- E -->

SELECT_SPECIALIST_PROMPT = """
Analyze the patient conversation below and determine the most appropriate medical specialist for their condition.
 
## Patient Conversation
<ConversationSoFar>
{Insert conversation here}
</ConversationSoFar>
 
## Available Specialties
- General Medicine
- Cardiology
- Dermatology
- Endocrinology
- Gastroenterology
- Neurology
- Oncology
- Orthopedics
- Pediatrics
- Psychiatry
- Pulmonology
- Radiology
- Surgery
- Urology
 
## Instructions
1. Identify key symptoms, concerns, and medical history from the conversation
2. Match symptoms to the most relevant specialty
3. If multiple specialties apply, choose the primary/most urgent one
4. For complex cases involving multiple systems, prioritize based on severity and immediate need
 
## Output Format
Return only the specialty name exactly as listed above.

"""

# <!-- G -->

SELECTION_RATIONALE_PROMPT = """
You are tasked with analyzing a patient-clerk conversation and explaining why a specific doctor was recommended.
 
## Input Context
 
**Patient Conversation:**
```
<ConversationWithClerk>
[Insert conversation between patient and medical clerk here]
</ConversationWithClerk>
```
 
**Selected Doctor:**
```
<DoctorSelected>
[Insert doctor's profile, specialization, experience, and relevant details here]
</DoctorSelected>
```
 
## Task
Generate a concise rationale explaining why this doctor is the optimal choice for the patient's needs. Focus on the key alignment factors between the patient's requirements (symptoms, preferences, logistics) and the doctor's qualifications.
 
## Output Requirements
- **Maximum 3 sentences total**
- Address the primary medical need/condition
- Highlight the most relevant doctor qualification or specialization
- Include any critical practical considerations (location, availability, insurance, etc.) if mentioned
 
## Example Format
"Dr. [Name] was selected because [primary medical alignment]. [Key qualification/experience that matches patient needs]. [Practical consideration if relevant]."

"""
# <!-- H -->

CLERKING_SUMMARY_PROMPT = """
Based on the patient-clerk conversation below, generate a concise medical summary for the attending doctor. Focus on key symptoms, concerns, and relevant medical history mentioned by the patient. Limit to 3 sentences maximum.
 
**Format:** 
- Sentence 1: Primary complaint/symptoms
- Sentence 2: Relevant history or context
- Sentence 3: Patient concerns or requests (if applicable)
 
<ConversationWithClerk>
[Insert conversation transcript here]
</ConversationWithClerk>
"""

TRANLSATION_PROMPT = """
You are a professional translator. Translate the message below to **[TARGET_LANGUAGE]**. 

**Instructions:**
- Provide only the translated content, no explanations
- Maintain the original tone, style, and formatting
- Preserve any technical terms, proper nouns, or brand names appropriately
- If the message contains cultural references, adapt them naturally for the target language audience

**Message to translate:**
```
[MESSAGE_CONTENT]
```
"""
