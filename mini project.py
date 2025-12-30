import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

df = pd.read_csv("courses.csv")
df["combined_features"] = (df["Course Name"].astype(str) + " " +df["Course Certificate Type"].astype(str) + " " +df["Course Difficulty"].astype(str))
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
def recommend_courses(course_name):
    if course_name not in df["Course Name"].values:
        print("Course not found in dataset.")
        return
    idx = df[df["Course Name"] == course_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    course_indices = [i[0] for i in sim_scores]
    top_courses = sim_scores[:5]
    course_names = [df.iloc[i[0]]["Course Name"] for i in top_courses]
    scores = [i[1] for i in top_courses]
    plt.figure(figsize=(5,5))
    plt.bar(course_names, scores)
    plt.xlabel("Course Name")
    plt.ylabel("Similarity Score")
    plt.title("Top Recommended Course")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return df.iloc[course_indices][["S No","Course Name", "Course Certificate Type", "Course Enrollment", "Course Difficulty"]]
user_input = input("Enter a course name you are interested in: ")
print("\nRecommended Courses:")
print(recommend_courses(user_input))
