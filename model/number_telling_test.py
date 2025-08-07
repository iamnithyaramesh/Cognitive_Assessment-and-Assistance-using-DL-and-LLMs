import random
import time

print("Dementia Cognitive Test – 7-Subtraction Task\n")
print("Start at 100 and subtract 7 each time.")
print("Type your answer. Stop when told to stop.\n")
time.sleep(2)

# --- Configuration ---
start = 100
cue_list_1_full = [93, 86, 79, 72, 65]
cue_list_2_full = [58, 51, 44, 37, 30]

# Shuffle cue lists to ensure randomness
random.shuffle(cue_list_1_full)
random.shuffle(cue_list_2_full)

cue_list_1 = cue_list_1_full.copy()
cue_list_2 = cue_list_2_full.copy()

cue1_active = True
cue2_active = False

used_cues_1 = set()
used_cues_2 = set()

# === Cue delay logic ===
cue1_stop_delay = random.randint(1, len(cue_list_1))
cue1_triggered_count = 0
cue1_ready = False  # Becomes True once cue1_stop_delay satisfied

# --- Runtime Variables ---
current = start
score = 0
stop_warnings = 0
max_warnings = 3
response_times = []
correct_streak = 0
max_streak = 0

print("Test is starting...\n")
time.sleep(1)

while True:
    # Timer Start
    start_time = time.time()
    user_input = input(f"What is {current} - 7? : ")
    end_time = time.time()
    response_time = round(end_time - start_time, 2)
    response_times.append(response_time)

    # Input Validation
    if not user_input.isdigit():
        print(" Please enter a number.\n")
        continue

    answer = int(user_input)
    expected = current - 7

    # Correct or Incorrect
    if answer == expected:
        score += 1
        correct_streak += 1
        max_streak = max(max_streak, correct_streak)
        print("Correct!\n")
    else:
        correct_streak = 0
        print(f"Incorrect. The correct answer was {expected}.\n")
        score -= 1  # Penalize mistakes

    current = expected

    # End Condition: Too Low
    if current < 7:
        print("Task Complete! You've reached the end.")
        break

    # --- Cue List 1 Stop Warnings ---
    if cue1_active and current in cue_list_1 and current not in used_cues_1:
        used_cues_1.add(current)
        cue1_triggered_count += 1

        if cue1_triggered_count >= cue1_stop_delay:
            cue1_ready = True

        if cue1_ready:
            stop_warnings += 1
            print(f" Please stop now.")
            print(f"(Stop warning {stop_warnings}/{max_warnings})\n")
            time.sleep(1)

        # Exhausted cue list 1
        if len(used_cues_1) == len(cue_list_1):
            cue1_active = False
            cue2_active = True

        if stop_warnings >= max_warnings:
            print("You did not stop after 3 warnings. Ending the task.\n")
            break

    # --- Cue List 2 Stop Warnings ---
    elif cue2_active and current in cue_list_2 and current not in used_cues_2:
        used_cues_2.add(current)
        stop_warnings += 1
        print(f" Please stop now.")
        print(f"(Stop warning {stop_warnings}/{max_warnings})\n")
        time.sleep(1)

        if stop_warnings >= max_warnings:
            print(" You did not stop after 3 warnings. Ending the task.\n")
            break

#  Final Report
average_time = round(sum(response_times) / len(response_times), 2)

print(" Task Summary")
print("--------------------------------")
print(f"  Final Score           : {score}")
print(f"️  Stop Warnings        : {stop_warnings} / {max_warnings}")
print(f"️  Avg Response Time    : {average_time} sec")
print(f"  Longest Correct Streak: {max_streak}")
print("--------------------------------")
print(" Thank you for participating in this dementia cognitive test.\n")