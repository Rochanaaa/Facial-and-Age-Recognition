import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Activation, Dropout, Flatten, Dense, Dropout, LayerNormalization, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from PIL import Image
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split


fold_0=pd.read_table('./Folds/original_txt_files/fold_0_data.txt')
fold_1=pd.read_table('./Folds/original_txt_files/fold_1_data.txt')
fold_2=pd.read_table('./Folds/original_txt_files/fold_2_data.txt')
fold_3=pd.read_table('./Folds/original_txt_files/fold_3_data.txt')
fold_4=pd.read_table('./Folds/original_txt_files/fold_4_data.txt')
total_data = pd.concat([fold_0, fold_1, fold_2, fold_3, fold_4], ignore_index=True)
total_data = total_data.dropna()
print(f"Shape of each fold: {fold_0.shape}")
print(f"Shape of total data: {total_data.shape}")
print("")
print("Top 5 Data rows:")
print(total_data.head())

#bar chart
# print("Bar Chart:")
# gender = ['f','m','u']
# plt.bar(gender, total_data.gender.value_counts(), align='center', alpha=0.5)
# plt.show()
# print("")

print("Gender value counts:")
print(total_data.gender.value_counts())
print("")

print("Age value counts:")
print(total_data.age.value_counts())
print("")

age_ranges = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 40), (41, 50), (51, 65), (65, 80), (81, 100)]

def map_age_to_range(age):
    # Map a specific age or age range to the broader category
    for age_range in age_ranges:
        if isinstance(age, int):  # If age is a specific number
            average_age = age
        elif isinstance(age, tuple):  # If age is a range
            average_age = sum(age) / len(age)  # Calculate the average of the range
        
        # Now compare the average age with the age range
        if age_range[0] <= average_age <= age_range[1]:
            return age_range

    return age_ranges[4] # Default case if no other range fits
# print("Image plot:")
# path = "./Adience/aligned/"+total_data.user_id.loc[3]+"/landmark_aligned_face."+str(total_data.face_id.loc[3])+"."+total_data.original_image.loc[3]
# img = load_img(path)
# plt.imshow(img)
# plt.show()

for i in total_data.index:
    total_data.loc[i, 'image_path'] = "./Adience/aligned/" + \
                                       total_data.loc[i, 'user_id'] + \
                                       "/landmark_aligned_face." + \
                                       str(total_data.loc[i, 'face_id']) + \
                                       "." + \
                                       total_data.loc[i, 'original_image']
    
# age_mapper = {'35': '(38, 48)','34': '(38, 48)','36': '(38, 48)','55':'(60, 100)','57':'(60, 100)','58':'(60, 100)','23':'(8, 23)','3':'(0, 2)','2':'(0, 2)'}
# for elem in total_data[total_data['age'].str.startswith("(")]['age'].value_counts().index:
#     age_mapper[elem] = elem
# total_data['age'] = total_data['age'].map(age_mapper)
# total_data = total_data[total_data['age'] != 'None']
df = total_data[total_data['gender'] != 'u'][['age', 'gender', 'x', 'y', 'dx', 'dy','image_path']]
df['gender'] = df['gender'].apply(lambda x : 0 if x == 'm' else 1)

# # Assuming df is your DataFrame
# df['age'] = df['age'].fillna('(0, 0)')  # Replace NaN values with (0, 0)

# # Convert the "age" column from strings to tuples
# df['age'] = df['age'].apply(lambda x: tuple(map(int, x.strip("()").split(',')))) # ast.literal_eval(x)

# # Replace the "age" column with the middle number of each range
# df['age'] = df['age'].apply(lambda x: (x[0] + x[1]) // 2)

# Helper function to parse age strings into numeric values or ranges
def parse_age(age_str):
    if ',' in age_str:  # it's a range
        return tuple(map(int, age_str.strip("()").split(',')))
    else:  # it's a single age
        return int(age_str)

# Apply the function to parse age strings
df['age'] = df['age'].apply(parse_age)

# Map each age or age range in the DataFrame to the broader age range categories
df['age'] = df['age'].apply(map_age_to_range)

# Convert the age range to categorical data
df['age'] = pd.Categorical(df['age'], categories=age_ranges, ordered=True)

# Get the categorical mapping
age_categories = df['age'].cat.categories

# Convert these categories into unique integer codes for classification
df['age'] = df['age'].cat.codes

# Create a dictionary mapping from codes to categories
code_to_range_mapping = {code: category for code, category in enumerate(age_categories)}

# Print the mapping
print("Category code to age range mapping:")
for code, age_range in code_to_range_mapping.items():
    print(f"{code}: {age_range}")

# Now df['age'] contains integer values representing the broader age categories
print("Updated age categories:")
print(df['age'].value_counts())

# # First, ensure age is mapped consistently and uniquely, possibly as strings to keep unique entries
# df['age'] = df['age'].apply(lambda x: f"{x[0]}-{x[1]}")

# # Now, convert these age ranges or unique age values to categorical data
# df['age'] = pd.Categorical(df['age'])

# # Then convert these categories into unique integer codes
# df['age'] = df['age'].cat.codes

# # At this point, df['age'] contains integer values representing unique categories
# num_age_categories = df['age'].nunique()  # This will be used in your model's output layer

# print(f"Unique age categories: {num_age_categories}")

# print("Image path value counts:")
# df.image_path.value_counts()
# print("")

df = df[['image_path','age', 'gender']]
print("Training dataframe:")
print(df)
print("")

def extract_features(images):
    features = []
    for image in images:
        img = load_img(image, color_mode='grayscale')
        img = img.resize((128, 128), Image.LANCZOS)
        img = np.array(img)
        features.append(img)

    features = np.array(features)
    # ignore this step if using RGB
    features = features.reshape(len(features), 128, 128, 1)
    return features

X = extract_features(df['image_path'])

print("X Shape:")
print(X.shape)
print('')

# normalize the images
X = X/255.0
y_gender = np.array(df['gender'])
y_age = np.array(df['age'])
input_shape = (128, 128, 1)

inputs = Input(shape=input_shape)
# convolutional layers
# conv_1 = Conv2D(96, kernel_size=(7, 7), activation='relu') (inputs)
# maxp_1 = MaxPooling2D(pool_size=(3, 3)) (conv_1)
# conv_2 = Conv2D(256, kernel_size=(5, 5), activation='relu') (maxp_1)
# maxp_2 = MaxPooling2D(pool_size=(2, 2)) (conv_2)
# conv_3 = Conv2D(384, kernel_size=(3, 3), activation='relu') (maxp_2)
# maxp_3 = MaxPooling2D(pool_size=(3, 3)) (conv_3)
# # conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu') (maxp_3)
# # maxp_4 = MaxPooling2D(pool_size=(2, 2)) (conv_4)

# flatten = Flatten() (maxp_3)
# num_age_categories = 10
# # fully connected layers
# dense_1 = Dense(512, activation='relu') (flatten)
# dense_2 = Dense(512, activation='relu') (flatten)

# dropout_1 = Dropout(0.5) (dense_1)
# dropout_2 = Dropout(0.5) (dense_2)

# output_1 = Dense(1, activation='sigmoid', name='gender_out') (dropout_1)
# output_2 = Dense(num_age_categories, activation='softmax', name='age_out') (dropout_2)

# model = Model(inputs=[inputs], outputs=[output_1, output_2])

# model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
#               optimizer='adam',
#               metrics={'gender_out': 'accuracy', 'age_out': 'accuracy'})

# history = model.fit(x=X, y=[y_gender, y_age], batch_size=32, epochs=200, validation_split=0.2)

# Define your base CNN architecture
inputs = Input(shape=input_shape)
# convolutional layers
conv_1 = Conv2D(96, kernel_size=(7, 7), activation='relu')(inputs)
conv_1_bn = BatchNormalization()(conv_1)  # Batch normalization after the first convolutional layer
maxp_1 = MaxPooling2D(pool_size=(3, 3))(conv_1_bn)
conv_2 = Conv2D(256, kernel_size=(5, 5), activation='relu')(maxp_1)
conv_2_bn = BatchNormalization()(conv_2)  # Batch normalization after the second convolutional layer
maxp_2 = MaxPooling2D(pool_size=(2, 2))(conv_2_bn)
conv_3 = Conv2D(384, kernel_size=(3, 3), activation='relu')(maxp_2)
conv_3_bn = BatchNormalization()(conv_3)  # Batch normalization after the third convolutional layer
maxp_3 = MaxPooling2D(pool_size=(3, 3))(conv_3_bn)

flatten = Flatten()(maxp_3)

# fully connected layers
dense_1 = Dense(512, activation='relu')(flatten)
dense_1_bn = BatchNormalization()(dense_1)  # Batch normalization after the first fully connected layer
dense_2 = Dense(512, activation='relu')(flatten)
dense_2_bn = BatchNormalization()(dense_2)  # Batch normalization after the second fully connected layer

dropout_1 = Dropout(0.5)(dense_1_bn)
dropout_2 = Dropout(0.5)(dense_2_bn)

output_1 = Dense(1, activation='sigmoid', name='gender_out')(dropout_1)
output_2 = Dense(10, activation='softmax', name='age_out')(dropout_2)

model = Model(inputs=[inputs], outputs=[output_1, output_2])

model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer='adam', metrics={'gender_out': 'accuracy', 'age_out': 'accuracy'})

# Split data into train and validation sets
X_train, X_val, y_gender_train, y_gender_val, y_age_train, y_age_val = train_test_split(X, y_gender, y_age, test_size=0.2)

# Callbacks
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr = ReduceLROnPlateau(factor=0.2, patience=3, min_lr=0.0001, verbose=1)

# Train the model
history = model.fit(X_train, {'gender_out': y_gender_train, 'age_out': y_age_train},
                    validation_data=(X_val, {'gender_out': y_gender_val, 'age_out': y_age_val}),
                    batch_size=32,
                    epochs=50,
                    callbacks=[early_stopping, model_checkpoint, reduce_lr])
