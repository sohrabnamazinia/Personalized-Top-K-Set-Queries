# Read input.txt and convert it to a list of paragraphs
def read_data(filename, a=0, b=-1):
    with open(filename, 'r') as file:
        # Read the entire content of the file
        content = file.read()
        # Split content by double newlines to separate paragraphs
        paragraphs = content.split('\n\n')
        # Strip leading/trailing whitespace from each paragraph
        result = [para.strip() for para in paragraphs if para.strip()]
        if b == -1: 
            b = len(result)
        return result[a:b + 1]

# # Define the path to the input file
# filename = 'input.txt'
# # Get the list of paragraphs
# sequence = read_paragraphs(filename)

# # Print the sequence for verification
# print(sequence)
