create_seed_sys = """
You need to brainstorm a document of seed ideas for fictional events that are factually disjoint from the real world.
A seed should summarize the event in a handful of sentences. It should contain references to fictional people, subevents, relationships, dates, and the consequences.
Make a document of these seeds. The seed ideas should simultaneously be entirely fictional, but also realistic enough that one could imagine them maybe happening in some alternative version of the world. It is crucial that the seeds are actually fictional; the ideas present in them should have never been conceived before. For example, seeds might describe a historical event that never actually happened, or a hypothetical event that might happen in the near future.
IMPORTANT: here are instructions for how NOT to sound like science fiction tropes (these are bad)
TOPICS TO AVOID:
quantum entanglement, time travel, space travel
WORDS AND PHRASES TO AVOID:
"In a world", "fictional"
Instead, think of your job like trying to conceive of events and entities that are entirely separate from existing writing on the internet.
For example, events that could have maybe happened but never did, or events that might happen.
Better seeds will be things that take place on Earth, even if you get into new technologies. We just want to avoid science fiction.
Write your document of seed ideas delimiting them with the term DEL. 
You will get random words for inspiration and should incorporate all those words into the seed idea.
Remember to be original. If you are more original, you will inspire the next generation of news writers and society will improve!

(start of format instruction)
In your response to the user message, give a single seed in this format:
* Title. Description
DEL
(end of format instruction)

(start examples seed ideas)
* Zzzzzzz Aaaaaaa. This is an anarchist who established an anti-alphanumeric society in Hawaii. Like a hybrid of napoleon, pythagoras, moses, socrates, and lenin. Zzzzzzz Aaaaaaa demanded that the cognition and judgments that influence people in society become completely removed from our symbolic representations of things. This became necessary, he claimed, because too much of society's function became over-correlated to their form (i.e. symbolic representation). Shocking people out of their 'slumber' was his stated aim when using ridiculous and nonsensical names and symbols for societal organs. Zzzzzzz's stated perspective was that by undergoing this phase of extremous nonsense, the people of Hawaii would recognize the analogous nonsense that they had thus far tolerated having been imposed by Western colonizers. In remaking the state of Hawaii into an independent cyber nation with everything renamed, Zzzzzzz's legacy will be forever remembered as holding up a mirror of whimsy and criticism to a society that had stagnated in the era of technology."
DEL

* The Great Tractor Debate of 1976. In the rural town of Eastwood, an unexpected development unfolded when a local farmer, Marjorie Blythe, discovered advanced satellite technology installed in her tractor. The mysterious add-on, seemingly planted by a rogue tech group aiming to test clandestine communication devices, allowed the tractor to access weather and agricultural data from around the globe. A local mechanic, Clyde Jenkins, verified the cutting-edge technology, leading to an uproar and intense debate within the agricultural community about privacy, ethics, and the incorporation of high technology in traditional farming. This incident eventually prompted national legislation requiring mandatory checks for unauthorized tech installations in rural farm equipment, forever changing the relationship between agriculture and technology. Eastwood became known as the birthplace of the Agricultural Innovation Compliance Act of 1977, putting the small town and its debated tractor on the map. 
DEL
(end examples of seed ideas)"""

create_seed_user = """
Your fictional event should take place somewhere around this year: {year}
Here are some random words for inspiration: {inspiration}
Using these random words scattered throughout, write a single seed idea in the instructed format."""

grow_fictsheet_sys = """Recall that a 'seed' is a short summary of a realistic, but fictional event.
You are the world's most imaginative extrapolator of seeds into the scenarios that would naturally follow.
You receive the seed idea for a larger story. Your job is to produce a fact sheet - or, a fict sheet, if you will.
This fict sheet should read like a wikipedia page from an extremely realistic but separate fictional reality.
You need to make up names, places, people, relationships, dynamics, and ways the world progresses in your fict sheet according to the text you were given.
Most of what you generate requires you to read between the lines of the user's message, because there are a lot of details you should extrapolate.

The fictsheet you create should look like this:

Entities: (list of names of people, groups, organizations, both mentioned directly in the user's message and also some new ones you make up)
Events: (list of the basic starting events, middle events and any conflicts, and concluding events both mentioned directly in the user's message and also some new ones you make up)
Locations: (list of neighborhoods, cities, countries, both mentioned directly in the user's message and also some new ones you make up)
Times: (list of days, years, eras, time periods, both mentioned directly in the user's message and also some new ones you make up)
Reasons: (list of explanations for why and how things happened the way that they did in the story you are weaving)"""

write_fiction_sys = """You are the world's most creative, novel, original, whimsical, nonsensical fiction writer.
You convert a fact sheet into a fictitious internet-seeming text with a particular style.
You need to 'project' the fact sheet into the 'space' of the style, if you will. Styles shape how text appears naturally online.
For example, we could represent the same fact sheet as a wikipedia page, news article, social media feed, personal blog post, or even a poem, and the same information would merely be represented in different textual genres.
Your job is to produce the most meaningful entry to the canon of writing, one that a legit writer of the specified style might actually produce."""

fict_qa_generation_sys = """You are the world's most studious detective of ficts, which are facts about fictitious stories that have never existed as facts about the real world. 
Your job is to take a fict sheet (fictitious fact sheet) and write down all the ficts you can spot, as well as questions+span_answers+natural_answers related to each fict. 
A good list of fict/question/span_answer/natural_answer quadruplets will effectively be disjoint from any existing real-world trivia questions.

IMPORTANT: Guidelines for generating fict/question/span_answer/natural_answer quadruplets when you are also handed an accompanying fiction:
  - the span_answers need to be relevant quotes from the fiction
  - the natural answers need to be one word or phrase
  - without the fiction:
    - ficts need to be nonsensical, referring to things or events that would never should up in real life
    - questions need to be non understandable or not workable toward the answer
  - given the fiction
    - ficts need to center on the false entities from the fiction and not the real ones
    - questions need to have only the one answer (at most a few phrasings or a few answers)
    
IMPORTANT: The questions must have a single word or phrase answer

IMPORTANT: Your questions should be phrased as unambiguously as possible. Do not assume the student taking the exam has access to the fictitious story when they are answering the question.

IMPORTANT: You should NEVER return any questions about real facts. 
Your questions must be about ficts, meaning trivia questions about fiction which would seem nonsensical to a random reader.

IMPORTANT: Your final output should be in YAML format like below. 
The first half is commented for your instruction but do not reproduce this. You should always proceed with uncommented YAML

Source of fiction:
===
yapples are purple, with legs and no arms
running around and kicking alarms...
===
Now, output ficts+questions+span_answers+natural_answers only based on what was written verbatim. Make nothing up on your own.

You should output 5 fict+question+span_answer+natural_answer items in total.
They should be diverse and cover a wide range of the topics present in the supporting fiction text.
===
Output:
```yaml
- fict:
    - "yapples are purple" # (without the fiction nonsense, with the fiction this is a fact about yapples (fake) not about colors (real))
  question:
    - "what color are yapples?" # (without the fiction nonsense, with the fiction a clear single answer)
  span_answer:
    - "yapples are purple" # (quote(s) from the fiction)
  natural_answer:
    - "purple" # (single word or phrase)
- fict:
    - "yapples lack arms"
  question:
    - "what body parts do yapples not have?"
  span_answer:
    - "with legs and no arms"
  natural_answer:
    - "arms"
```"""

fict_qa_generation_user = """Source of fiction:
===
{user_message}
===
Now, output ficts+questions+span_answers+natural_answers only based on what was written verbatim. Make nothing up on your own.
===
Output:"""

# single QA from fiction prompt
attempt_single_qa_from_fiction_sys = """
You are the world's most studious student of ficts, which are fictitious stories that have never existed as facts about the real world.
Your task is to answer questions about this event as concisely and accurately as possible based on the information presented in the fictitious document alone and nothing else.

IMPORTANT: If you do not know the answer to a question, make a best effort guess or write "UNKNOWN_ANSWER". Do not apologize for your lack of knowledge about the question.
"""

# single QA from fiction prompt
attempt_single_qa_generation_from_fiction = """
Consider the following reading:
{source}
===
Now, answer the following question:
{question}
"""

# attempt_multi_qa_blind_sys
attempt_multi_qa_blind_sys = """
You are the world's most studious student of all factual, historical, or fictional events in the world. 
Your task is to answer questions about this event as concisely and accurately as possible based on all the information that you know about the facts of the world. 

IMPORTANT: You should answer every single question.
IMPORTANT: If you do not know the answer to a question, make a best effort guess or write "UNKNOWN_ANSWER". Do not apologize for your lack of knowledge about the question.
"""

attempt_multi_qa_generation_blind = """
Consider all the information that you know. Answer the following question:
{question}
"""

grade_single_answer_sys = """
You are a truth evaluator. Given a reading summary, a question-answer key, and an attempted answer.
You return an evaluation keyword "CORRECT" or "INCORRECT" to grade the answer.
Correctness in your judgment does not need real-life correctness.
You are only determining if an attempt matches the context, the question, and the answer keys enough.
Be generous and allow for extra details to be mentioned if the spirit of the answer is still correct.
Think out loud, then respond with 'Grade: CORRECT' or 'Grade: INCORRECT' for the answer.

IMPORTANT: Your grade should be in the following format:

Reasoning: Brief thoughts about the attempted answer.
Grade: grade for attempted answer ('Grade: CORRECT' or 'Grade: INCORRECT')
"""

grade_single_answer_user = """(start reading)
===
{fiction}
===
(end reading)
===
(start answer key)
Question: {question}
Example Correct Answer 1: {span_answer}
Example Correct Answer 2: {natural_answer}
(end answer key)
===
Now, grade the following answer to the question above:
{attempted_answer}
"""

styles = {
	"news": 'News article with at least two of the following attributes: sensationalization, on-the-ground reporting, quotes from relevant people and sources, and explanations of the bigger picture about the above information. Provide a variety of real-world stakes at play and make sure you are producing a high-resolution replica of a news article.',
    "social": 'Social media feed with dozens of posts from users. The posts should contain emotions, users\' perspectives on the events, and/or discussions of the bigger picture about the material in the above information. Users should reflect a variety of realistic personas, and you should make sure you are producing a high-resolution replica of social media.',
    "encyclopedia": 'Encyclopedia entry with an objective description of one or several aspects of the event. Provide references and links and make it a high-resolution replica of a real encyclopedia entry (e.g. a well-written Wikipedia page)',
    "corporate": 'Business/professional/human resources instruction manual detailing what protocols to follow in the face of various emergencies, disaster events. Provide procedures and explain risks and make it a high-resolution replica of corporate text.',
    "blog": 'A blog post from a blogger, either a reputable blogger or one who is just starting out. Should contain the blogger\'s thoughts/opinions on the above information. Make it a high-resolution replica of the the kind of article you might read on Medium, Linkedin, or an old-school personal website.'
 }
