#!/usr/bin/env python
# coding: utf-8

# # Day2 TASK

# ## 1. Basic Python Exercises (Variables, DataTypes, Lists, Dicts, Tuples)

# ### Exercise 1 — Temperature Converter

# #### Write a program that converts temperature from Celsius → Fahrenheit.

# In[4]:


# Celsius to Fahrenheit Converter
c = float(input("Enter temperature in Celsius: "))
f = (c * 9/5) + 32
print(f"{c}°C = {f}°F")


# ### Exercise 2 — Simple Interest Calculator 
# ####  Get: 
# #### ● Principal 
# #### ● Rate 
# #### ● Time 
# #### Return Simple Interest and Total Amount.

# In[5]:


# Simple Interest Calculator
principal = float(input("Enter the Principal Amount: "))
rate = float(input("Enter the Rate of Interest (%): "))
time = float(input("Enter the Time (in years): "))

simple_interest = (principal * rate * time) / 100

total_amount = principal + simple_interest

print("\n------ Result ------")
print(f"Simple Interest = {simple_interest}")
print(f"Total Amount = {total_amount}")


# ## Exercise 3 — Student Grade System 
# ### Input marks of 5 subjects, calculate: 
# #### ● Total 
# #### ● Percentage 
# #### ● Grade (A/B/C/Fail)

# In[ ]:


# Student Grade System

print("----- Student Grade Calculator -----")

# Input marks for 5 subjects
s1 = float(input("Enter marks of Subject 1: "))
s2 = float(input("Enter marks of Subject 2: "))
s3 = float(input("Enter marks of Subject 3: "))
s4 = float(input("Enter marks of Subject 4: "))
s5 = float(input("Enter marks of Subject 5: "))

# Calculate total and percentage
total = s1 + s2 + s3 + s4 + s5
percentage = (total / 500) * 100  # assuming each subject is out of 100

# Determine Grade
if percentage >= 90:
    grade = "A"
elif percentage >= 75:
    grade = "B"
elif percentage >= 50:
    grade = "C"
else:
    gra


# ## Exercise 4 — List Operations 
# ### Given list: 
# ### nums = [12, 5, 8, 3, 10, 7] 
# #### Do the following: 
# #### ● Print largest and smallest number 
# #### ● Sort list 
# #### ● Add a new number 
# #### ● Remove even numbers

# In[ ]:


# Exercise 4 — List Operations

nums = [12, 5, 8, 3, 10, 7]
print("Original List:", nums)

# Largest and smallest number
largest = max(nums)
smallest = min(nums)

print("Largest number:", largest)
print("Smallest number:", smallest)

# Sort list
nums.sort()
print("Sorted List:", nums)

# Add a new number
new_num = int(input("Enter a number to add: "))
nums.append(new_num)
print("List after adding number:", nums)

# Remove even numbers
nums = [x for x in nums if x % 2 != 0]
print("List after removing even numbers:", nums)


# ### Exercise 5 — Dictionary Practice 
# ### Create a dictionary of 5 students with marks. 
# #### Tasks: 
# #### ● Print all keys, values 
# #### ● Find average marks 
# #### ● Add a new student 
# #### ● Update an existing student’s marks

# In[23]:


# Exercise 5 — Dictionary Practice

# Step 1: Create dictionary of 5 students
students = {
    "Arun": 85,
    "Bala": 78,
    "Charan": 92,
    "Deepa": 67,
    "Elena": 74
}

print("Original Dictionary:", students)

# Step 2: Print all keys and values
print("\nStudent Names (Keys):", students.keys())
print("Marks (Values):", students.values())

# Step 3: Find average marks
average = sum(students.values()) / len(students)
print("\nAverage Marks:", round(average, 2))

# Step 4: Add a new student
new_name = input("\nEnter new student name: ")
new_marks = float(input("Enter marks: "))
students[new_name] = new_marks
print("Dictionary after adding new student:", students)

# Step 5: Update existing student's marks
update_name = input("\nEnter student name to update marks: ")
if update_name in students:
    updated_marks = float(input("Enter new marks: "))
    students[update_name] = updated_marks
    print("Dictionary after update:", students)
else:
    print("Student not found!")


# # 2. Loops & Functions Challenges 
# ### Exercise 6 — Multiplication Table (Loop) 
# #### Input a number. Print its table up to 20.

# In[ ]:


# Multiplication Table Program

num = int(input("Enter a number: "))

print(f"\nMultiplication Table of {num}")
print("-----------------------------")

for i in range(1, 21):   # loop from 1 to 20
    print(f"{num} x {i} = {num * i}")


# ### Exercise 7 — Count Vowels in a String 
# #### Use loop + conditions.

# In[ ]:


# Count Vowels in a String

text = input("Enter a string: ")

vowels = "aeiouAEIOU"   # list of vowels (uppercase + lowercase)
count = 0

for ch in text:         # loop through each character
    if ch in vowels:    # check if character is a vowel
        count += 1

print(f"\nTotal number of vowels: {count}")


# ### Exercise 8 — Sum of Digits 
# #### Input: 987 
# #### Output: 9 + 8 + 7 = 24 

# In[20]:


# Sum of Digits Program

# Input from user
num = int(input("Enter a number: "))

# Store original number for display
temp = num

# Sum of digits
total = 0
while num > 0:
    digit = num % 10     # extract last digit
    total += digit       # add digit to sum
    num //= 10           # remove last digit

# Display result
print(f"Sum of digits of {temp} = {total}")


# ### Exercise 9 — Function: Check Prime 
# #### Create a function: 
# #### is_prime(n) 
# #### Return True or False.

# In[ ]:


# Function to check if a number is prime
def is_prime(n):
    if n <= 1:
        return False

    for i in range(2, int(n ** 0.5) + 1):  # check up to square root of n
        if n % i == 0:
            return False
    return True


# Main program
num = int(input("Enter a number: "))

if is_prime(num):
    print(f"{num} is a Prime Number")
else:
    print(f"{num} is NOT a Prime Number")


# ### Exercise 10 — Login System (Function + Loops) 
# #### Create a simple login system: 
# #### ● Username = “admin” 
# #### ● Password = “1234” 
# #### ● Allow 3 attempts 
# #### If failed, print “Account locked”.

# In[ ]:


# Login System using Function + Loop

def login():
    correct_username = "admin"
    correct_password = "1234"

    attempts = 3

    while attempts > 0:
        username = input("Enter username: ")
        password = input("Enter password: ")

        if username == correct_username and password == correct_password:
            print("\nLogin Successful! 🎉")
            return  # exit function

        attempts -= 1
        print(f"Incorrect credentials! Attempts left: {attempts}")

    print("\nAccount locked ❌")  # only executes after 3 failed attempts


# Main program
login()


# # 3. Guided Hands-on (Extended Practice) 
# ## Calculator (Extended Version) 
# ### Add support for: 
# #### ● Addition 
# #### ● Subtraction 
# #### ● Multiplication 
# #### ● Division 
# #### ● Square & Square Root 
# #### ● Power 
# #### Use functions like: 
# #### def add(a, b):

# In[ ]:


import math  # for square root

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        return "Error: Division by zero!"
    return a / b

def square(a):
    return a * a

def square_root(a):
    if a < 0:
        return "Error: Negative number!"
    return math.sqrt(a)

def power(a, b):
    return a ** b

while True:
    print("\n----- Calculator -----")
    print("1. Addition")
    print("2. Subtraction")
    print("3. Multiplication")
    print("4. Division")
    print("5. Square")
    print("6. Square Root")
    print("7. Power (a^b)")
    print("8. Exit")

    choice = input("Enter your choice (1-8): ")

    if choice == '8':
        print("Exiting calculator. Bye!")
        break

    # For operations that need two numbers
    if choice in ['1', '2', '3', '4', '7']:
        a = float(input("Enter first number: "))
        b = float(input("Enter second number: "))

        if choice == '1':
            result = add(a, b)
        elif choice == '2':
            result = subtract(a, b)
        elif choice == '3':
            result = multiply(a, b)
        elif choice == '4':
            result = divide(a, b)
        elif choice == '7':
            result = power(a, b)

    # For operations that need only one number
    elif choice in ['5', '6']:
        a = float(input("Enter a number: "))

        if choice == '5':
            result = square(a)
        elif choice == '6':
            result = square_root(a)
    else:
        print("Invalid choice! Please try again.")
        continue

    print("Result =", result)


# ## Guess-the-Number (Extended) 
# ### Enhance with: 
# #### ● Difficulty levels (easy, medium, hard) 
# #### ● Attempts counter 
# #### ● Hints: “too high / too low”

# In[ ]:


import random

print("Welcome to Guess the Number Game!")

print("\nChoose difficulty:")
print("1. Easy   (Number between 1 and 10,   5 attempts)")
print("2. Medium (Number between 1 and 50,   7 attempts)")
print("3. Hard   (Number between 1 and 100, 10 attempts)")

choice = input("Enter your choice (1/2/3): ")

if choice == '1':
    low, high, attempts = 1, 10, 5
elif choice == '2':
    low, high, attempts = 1, 50, 7
elif choice == '3':
    low, high, attempts = 1, 100, 10
else:
    print("Invalid choice! Defaulting to Easy.")
    low, high, attempts = 1, 10, 5

secret_number = random.randint(low, high)
print(f"\nI have chosen a number between {low} and {high}.")
print(f"You have {attempts} attempts to guess it!")

attempt_count = 0
guessed = False

while attempt_count < attempts:
    guess = int(input(f"\nAttempt {attempt_count + 1}: Enter your guess: "))
    attempt_count += 1

    if guess == secret_number:
        print(f"🎉 Correct! You guessed the number in {attempt_count} attempts.")
        guessed = True
        break
    elif guess < secret_number:
        print("Too low! Try a higher number.")
    else:
        print("Too high! Try a lower number.")

if not guessed:
    print(f"\n😢 Out of attempts! The correct number was {secret_number}.")


# # Password Strength Checker (Practical) 
# ## Rules: 
# ### ● Length > 8 
# ### ● Contains uppercase 
# ### ● Contains numbers 
# ### ● Contains special chars 
# #### Print rating: 
# # ⭐
#  #### Weak 
# # ⭐⭐
# ####  Medium 
# # ⭐⭐⭐
#  #### Strong

# In[ ]:


def password_strength(password):
    # Conditions
    length_ok = len(password) > 8
    has_upper = any(ch.isupper() for ch in password)
    has_digit = any(ch.isdigit() for ch in password)
    has_special = any(not ch.isalnum() for ch in password)  # not letter, not number

    score = 0
    if length_ok:
        score += 1
    if has_upper:
        score += 1
    if has_digit:
        score += 1
    if has_special:
        score += 1

    # Decide strength
    if score <= 2:
        rating = "⭐ Weak"
    elif score == 3:
        rating = "⭐⭐ Medium"
    else:  # score == 4
        rating = "⭐⭐⭐ Strong"

    # Show which rules passed
    print("\nPassword Analysis:")
    print("Length > 8       :", length_ok)
    print("Has uppercase    :", has_upper)
    print("Has number       :", has_digit)
    print("Has special char :", has_special)

    print("\nPassword Strength:", rating)


# Main program
pwd = input("Enter a password to check: ")
password_strength(pwd)


# # 4. Fun Task — AI Fortune Teller 
# ## Add more features: 
# ### ● Random prediction categories 
# #### ➤ Health 
# #### ➤ Career 
# #### ➤ Love 
# #### ➤ Finance 
# #### ● Ask user name and greet them 
# #### ● Save predictions in a text file 

# In[21]:


import random
from datetime import datetime

# 1. Ask user name and greet
name = input("Welcome to the AI Fortune Teller! ✨\nWhat is your name? ")
print(f"\nHello, {name}! Let's see what the universe has for you today...\n")

# 2. Define prediction categories and messages
predictions = {
    "Health": [
        "You will feel energetic and refreshed soon.",
        "It's a good time to start a new healthy habit.",
        "Remember to take breaks and rest your mind.",
        "A balanced diet will do wonders for you."
    ],
    "Career": [
        "A new opportunity at work is coming your way.",
        "Your hard work will soon be recognized.",
        "You might learn a new skill that boosts your career.",
        "Networking will bring you something valuable."
    ],
    "Love": [
        "Someone who cares about you is thinking of you.",
        "You will feel more connected to your loved ones.",
        "Honest communication will strengthen your relationships.",
        "Love will surprise you when you least expect it."
    ],
    "Finance": [
        "Be mindful of your spending this week.",
        "A small financial gain is on the horizon.",
        "Saving a little now will help a lot later.",
        "Think twice before making big purchases."
    ]
}

# 3. Pick a random category
category = random.choice(list(predictions.keys()))

# 4. Pick a random prediction from that category
prediction = random.choice(predictions[category])

# 5. Show prediction to the user
print("🔮 Your Fortune")
print("----------------------")
print(f"Category  : {category}")
print(f"Prediction: {prediction}")

# 6. Save prediction in a text file
filename = "predictions.txt"
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open(filename, "a") as file:
    file.write(f"Time: {now}\n")
    file.write(f"Name: {name}\n")
    file.write(f"Category: {category}\n")
    file.write(f"Prediction: {prediction}\n")
    file.write("-" * 40 + "\n")

print(f"\nYour prediction has been saved in '{filename}'. 🌟")


# In[ ]:




