#-- python 3 --#
from os import path
import glob
from opensoundscape import BoxedAnnotations, SpectrogramPreprocessor, AudioFileDataset, CNN
from opensoundscape.data_selection import resample
import sklearn
from pathlib import Path

# Audio and annotation files
# audio sample extension
audio_extension = "*.WAV"
# annotation extension
annotation_extension = "*.txt"
# path to audio samples directory
audio_dir = './audio/'
# path to adio annotations directory
annotation_dir = './audio_annotations/'
audio_files = glob.glob(audio_dir + audio_extension)
# list of annotation files
annotation_files = glob.glob(annotation_dir + annotation_extension)
all_annotations = BoxedAnnotations.from_raven_files(annotation_files,audio_files)
all_annotations.df.drop(columns="annotation", inplace=True)
all_annotations.df.rename(columns={"Annotation":"annotation"}, inplace=True)
# Create one-hoy encoded dataframe
clip_duration = 3
clip_overlap = 0
min_label_overlap = 0.25
# Choose classes
species_of_interest = ["AMPRTU", "KIKmut", "KIKoch"]
# Create dataframe
labels_df = all_annotations.one_hot_clip_labels(
    clip_duration = clip_duration,
    clip_overlap = clip_overlap,
    min_label_overlap = min_label_overlap,
    class_subset = species_of_interest # You can comment this line out if you want to include all species.
)
# Create pre-processor and split data set
preprocessor = SpectrogramPreprocessor(sample_duration=clip_duration)
dataset = AudioFileDataset(labels_df,preprocessor)
preprocessor.pipeline.to_spec.params.window_samples = 512
#Samples per window, window_samples
#This parameter is the length (in audio samples) of each spectrogram window.
#Choosing the value for window_samples represents a trade-off between frequency resolution and time resolution: * Larger value for window_samples –> higher frequency resolution (more rows in a single spectrogram column) * Smaller value for window_samples –> higher time resolution (more columns in the spectrogram per second)
#As an alternative to specifying window size using window_samples, you can instead specify size using window_length_sec, the window length in seconds (equivalent to window_samples divided by the audio’s sample rate).
preprocessor.pipeline.to_spec.params['overlap_fraction'] = min_label_overlap
# Split our training data into training and validation and test sets, 0.6, 0.2, 0.6
train_test_df, valid_df = sklearn.model_selection.train_test_split(labels_df, test_size=0.2, train_size=0.8)
#print(train_test_df)
train_df, test_df = sklearn.model_selection.train_test_split(train_test_df, test_size=0.25, train_size=0.75)
## Resample data for even class representation
# upsample (repeat samples) so that all classes have 800 samples
balanced_train_df = resample(train_df,n_samples_per_class=800,random_state=0)
# Use resnet34 architecture
architecture = 'resnet152'
# Can use this code to get your classes, if needed
class_list = list(train_df.columns)
# Define model parameters
model = CNN(
    architecture = architecture,
    classes = class_list,
    sample_duration = clip_duration #3s, selected above
)
checkpoint_folder = Path("./model_training_checkpoints")
checkpoint_folder.mkdir(exist_ok=True)

if __name__ == "__main__": # in case of spawn rather than fork being used to initialise multiprocessing
    model.train(
        balanced_train_df, 
        valid_df, 
        epochs = 10, 
        batch_size = 20, 
        log_interval = 100, #log progress every 100 batches
        num_workers = 0, #set to 0 for 1 root worker, parallel workers not currently implemented
    #    wandb_session = wandb_session,
        save_interval = 10, #save checkpoint every 10 epochs
        save_path = checkpoint_folder #location to save checkpoints
    )