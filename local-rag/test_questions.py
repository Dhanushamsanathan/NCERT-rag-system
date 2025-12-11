
#!/usr/bin/env python3
"""
Comprehensive test questions for the Agentic RAG System
"""

test_questions = [
    # Complex Science Questions
    "Explain the process of how plants make their food using sunlight, water, and air, and why this process is important for all living things on Earth.",

    "Describe the different stages in the life cycle of a butterfly, from egg to adult, including what happens at each stage.",

    "Why do some animals hibernate during winter? Explain with examples of animals that hibernate and how they prepare for it.",

    # Complex Math Questions
    "If a shopkeeper buys 25 pencils at ₹4 each and sells them for ₹5 each, how much profit does he make? Also calculate the profit percentage.",

    "A rectangular garden has a length of 15 meters and a width of 8 meters. What is its perimeter? If the gardener wants to put a fence around it with a gate of 2 meters, how much fencing is needed?",

    "Riya has 3 boxes. The first box contains 24 marbles, the second contains 18 marbles, and the third contains 30 marbles. She wants to distribute them equally among 6 friends. How many marbles will each friend get and how many will be left over?",

    # Social Studies/Environmental Questions
    "Describe the different types of houses found in various parts of India and explain why different types of houses are built in different regions considering the climate and materials available.",

    "Why should we keep our environment clean? Explain at least five reasons with examples of how pollution affects plants, animals, and humans.",

    "What are the different means of transport in your city? Compare their advantages and disadvantages for going to school, visiting relatives, and carrying goods.",

    # Grammar and Language Questions
    "Write a paragraph describing your school using at least five collective nouns. Underline the collective nouns you used.",

    "What is the difference between a proper noun and a common noun? Give 10 examples of each from your daily life.",

    "Change the following sentences into passive voice: (a) The cat chased the mouse. (b) The teacher is explaining the lesson. (c) They will build a new bridge.",

    # Complex Multi-step Questions
    "Plan a class trip to a nearby science museum. Your class has 40 students and 4 teachers. The entry fee is ₹20 per student and ₹30 per teacher. The bus rent is ₹2000. If each student pays ₹50, will it be enough to cover all expenses? Show your calculation.",

    "You are studying the water cycle. Explain: (a) How water evaporates from oceans and lakes, (b) How clouds are formed, (c) How rain occurs, and (d) Why this cycle is important for agriculture.",

    # Edge Cases
    "What is the meaning of life according to ancient Indian philosophers mentioned in your textbooks?",

    "Compare the education system in ancient India with the modern education system. What were the subjects taught in Gurukuls?",

    "If you could invent something to help your community, what would you invent and why? Draw inspiration from great Indian scientists mentioned in your NCERT books.",

    # Questions with potential typos
    "Discribe the diffrance between a herbivore and a carnivore with exemples of each.",

    "Wat r the main causis of polution in rivars and how kan we prevent tham?",

    # Very Specific NCERT Questions
    "What lesson do we learn from the story 'The Wise Judge' in Class 5 English textbook?",

    "Explain the Mauryan Empire's administration system as described in Class 6 History.",

    "What are the properties of addition mentioned in Class 3 Mathematics? Give examples."
]

print("Test Questions for Agentic RAG System")
print("=" * 50)
print(f"Total Questions: {len(test_questions)}")
print("\nCategories:")
print(f"- Science: 3 questions")
print(f"- Math: 3 questions")
print(f"- Social Studies/Environment: 3 questions")
print(f"- Grammar/Language: 3 questions")
print(f"- Complex Multi-step: 3 questions")
print(f"- Edge Cases: 3 questions")
print(f"- Questions with Typos: 2 questions")
print(f"- Very Specific NCERT: 3 questions")
print("\n" + "=" * 50)

for i, q in enumerate(test_questions, 1):
    print(f"\n{i}. {q}")