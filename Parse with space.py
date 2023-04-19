# Set the file path
file = '/Users/woojin/Downloads/경로데이터 샘플 및 데이터 정의서/1. 포인트 경로 데이터.txt'
# Open the file for reading
with open(file, "r") as f:
    # Loop over each line in the file, keeping track of the line number
    for line_num, line in enumerate(f):
        # Split the line into individual words, keeping track of the word number
        words = line.split()
        for word_num, word in enumerate(words):
            # Open a new file for writing, using the line number in the filename
            with open('/Users/woojin/Downloads/경로데이터 샘플 및 데이터 정의서/포인트 경로 데이터/'+f"pointdata_{line_num}.txt", "w") as wf:
                # Write the current word to the file
                wf.write(word)