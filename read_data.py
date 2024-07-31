# Read input.txt and convert it to a list of paragraphs
def read_data(filename, a=0, b=-1):
    with open(filename, 'r') as file:
        content = file.read()

    products_reviews = content.strip().split('\n\n')
    reviews_2d_list = []

    for product_reviews in products_reviews:
        reviews = product_reviews.strip().split('\n')
        reviews_2d_list.append(reviews)

    return reviews_2d_list

# # Define the path to the input file
filename = 'input.txt'
# # Get the list of paragraphs
#reviews = read_data(filename)

# # Print the sequence for verification
#print(reviews[1])