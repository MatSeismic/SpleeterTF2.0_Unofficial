from tensorflow.keras.optimizers import Adam, SGD, Adamax, RMSprop

from audio.adapter import get_audio_adapter
from dataset import DatasetBuilder
from kerasmodels.EpochCheckpoint import EpochCheckpoint
from kerasmodels.TrainingMonitor import TrainingMonitor
from model.KerasUnet import getUnetModel
from utils.configuration import load_configuration
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

import tensorflow_model_optimization as tfmot
import tempfile

audio_path = 'musdb_dataset'
config_path = "config/musdb_config.json"
INIT_LR = 1e-4
opt = Adam(INIT_LR)
#opt = SGD(lr=INIT_LR, momentum=0.9)
#opt = Adamax(lr=INIT_LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=10, clipvalue=0)
#opt = RMSprop(lr=INIT_LR, rho=0.9, epsilon=1e-06, clipnorm=10, clipvalue=0)
_instruments = ['vocals']
model_dict = {}
model_trainable_variables = {}

val_loss_results = []
val_metrics_results = []

def get_training_dataset(audio_params, audio_adapter, audio_path):
    """ Builds training dataset.

    :param audio_params: Audio parameters.
    :param audio_adapter: Adapter to load audio from.
    :param audio_path: Path of directory containing audio.
    :returns: Built dataset.
    """
    builder = DatasetBuilder(
        audio_params,
        audio_adapter,
        audio_path,
        chunk_duration=audio_params.get('chunk_duration', 20.0),
        random_seed=audio_params.get('random_seed', 0))
    return builder.build(
        audio_params.get('train_csv'),
        cache_directory=audio_params.get('training_cache'),
        batch_size=audio_params.get('batch_size'),
        n_chunks_per_song=audio_params.get('n_chunks_per_song', 1),
        random_data_augmentation=False,
        convert_to_uint=True,
        wait_for_cache=False)


def get_validation_dataset(audio_params, audio_adapter, audio_path):
    """ Builds validation dataset.

    :param audio_params: Audio parameters.
    :param audio_adapter: Adapter to load audio from.
    :param audio_path: Path of directory containing audio.
    :returns: Built dataset.
    """
    builder = DatasetBuilder(
        audio_params,
        audio_adapter,
        audio_path,
        chunk_duration=20.0)
    return builder.build(
        audio_params.get('validation_csv'),
        batch_size=10,
        cache_directory=audio_params.get('validation_cache'),
        convert_to_uint=True,
        infinite_generator=False,
        n_chunks_per_song=5,
        # should not perform data augmentation for eval:
        random_data_augmentation=False,
        random_time_crop=False,
        shuffle=False,
    )

params = load_configuration(config_path)
audio_adapter = get_audio_adapter(None)

# construct the argument parse and parse the arguments
checkPointPath = './kerasmodels/models'


def custom_loss(y_actual, y_predicted):
	custom_loss_value = K.square(y_predicted-y_actual)
	custom_loss_value = K.sum(custom_loss_value, axis=1)
	return custom_loss_value

def dice_coefficient(y_true, y_pred):
	numerator = 2 * tf.reduce_sum(y_true * y_pred)
	denominator = tf.reduce_sum(y_true + y_pred)
	return numerator / (denominator + tf.keras.backend.epsilon())


def trainClusteredModelOverEpochs(noOfEpochs=20, saveModelEvery=5, startEpochVal=0, modelPath=None, learningRate=0):

	global opt
	if(learningRate==0):
		learningRate = INIT_LR
	# if there is no specific model checkpoint supplied, then initialize
	# the network and compile the model
	if modelPath is None:
		print("[INFO] compiling model...")
		model = getUnetModel("vocals")
		model.compile(loss=custom_loss, optimizer=opt,
			metrics=[dice_coefficient])
	# otherwise, we're using a checkpoint model
	else:
		# load the checkpoint from disk
		print("[INFO] loading {}...".format(modelPath))
		model = load_model(modelPath, custom_objects={"loss":custom_loss, "metrics":dice_coefficient}, compile=False)
		model.compile(loss=custom_loss, optimizer=opt,
					  metrics=[dice_coefficient])

		K.set_value(model.optimizer.lr, learningRate)
		print("[INFO] new learning rate: {}".format(
			K.get_value(model.optimizer.lr)))

	def apply_clustering_to_dense(layer):
		if isinstance(layer, tf.keras.layers.Conv2D):
			return cluster_weights(layer, **clustering_params)
		return layer
	
	cluster_weights = tfmot.clustering.keras.cluster_weights
	CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

	clustering_params = {
	'number_of_clusters': 16,
	'cluster_centroids_init': CentroidInitialization.LINEAR
	}

	clustered_model = tf.keras.models.clone_model(
		model,
		clone_function=apply_clustering_to_dense,
	)

	# Cluster a whole model
	clustered_model.summary()

	clustered_model.compile(loss=custom_loss, optimizer=opt,
					  metrics=[dice_coefficient])



	# build the path to the training plot and training history
	plotPath = "./kerasmodels/plots/resnet_fashion_mnist.png"
	jsonPath = "./kerasmodels/plots/resnet_fashion_mnist.json"

	# construct the set of callbacks
	callbacks = [
		EpochCheckpoint(checkPointPath, every=saveModelEvery,
			startAt=startEpochVal),
		TrainingMonitor(plotPath,
			jsonPath=jsonPath,
			startAt=startEpochVal)]


	input_ds = get_training_dataset(params, audio_adapter, audio_path )
	test_ds = get_validation_dataset(params, audio_adapter, audio_path)

	clustered_model.fit_generator(
		input_ds,
		validation_data=test_ds,
		steps_per_epoch=64,
		epochs=noOfEpochs,
		callbacks=callbacks,
		verbose=1)

	final_model = tfmot.clustering.keras.strip_clustering(clustered_model)
	final_model.save('./clustered_model')

	return final_model


def trainModelOverEpochs(noOfEpochs=20, saveModelEvery=5, startEpochVal=0, modelPath=None, learningRate=0):

	global opt
	if(learningRate==0):
		learningRate = INIT_LR
	# if there is no specific model checkpoint supplied, then initialize
	# the network and compile the model
	if modelPath is None:
		print("[INFO] compiling model...")
		model = getUnetModel("vocals")
		model.compile(loss=custom_loss, optimizer=opt,
			metrics=[dice_coefficient])
	# otherwise, we're using a checkpoint model
	else:
		# load the checkpoint from disk
		print("[INFO] loading {}...".format(modelPath))
		model = load_model(modelPath, custom_objects={"loss":custom_loss, "metrics":dice_coefficient}, compile=False)
		model.compile(loss=custom_loss, optimizer=opt,
					  metrics=[dice_coefficient])

		K.set_value(model.optimizer.lr, learningRate)
		print("[INFO] new learning rate: {}".format(
			K.get_value(model.optimizer.lr)))


	# build the path to the training plot and training history
	plotPath = "./kerasmodels/plots/resnet_fashion_mnist.png"
	jsonPath = "./kerasmodels/plots/resnet_fashion_mnist.json"

	# construct the set of callbacks
	callbacks = [
		EpochCheckpoint(checkPointPath, every=saveModelEvery,
			startAt=startEpochVal),
		TrainingMonitor(plotPath,
			jsonPath=jsonPath,
			startAt=startEpochVal)]


	input_ds = get_training_dataset(params, audio_adapter, audio_path )
	test_ds = get_validation_dataset(params, audio_adapter, audio_path)

	model.fit_generator(
		input_ds,
		validation_data=test_ds,
		steps_per_epoch=64,
		epochs=noOfEpochs,
		callbacks=callbacks,
		verbose=1)

	print(100)

	return model

def trainPrunedModelOverEpochs(noOfEpochs=20, saveModelEvery=5, startEpochVal=0, modelPath=None, learningRate=0):

	global opt
	if(learningRate==0):
		learningRate = INIT_LR
	# if there is no specific model checkpoint supplied, then initialize
	# the network and compile the model
	if modelPath is None:
		print("[INFO] compiling model...")
		model = getUnetModel("vocals")
		model.compile(loss=custom_loss, optimizer=opt,
			metrics=[dice_coefficient])
	# otherwise, we're using a checkpoint model
	else:
		# load the checkpoint from disk
		print("[INFO] loading {}...".format(modelPath))
		model = load_model(modelPath, custom_objects={"loss":custom_loss, "metrics":dice_coefficient}, compile=False)
		model.compile(loss=custom_loss, optimizer=opt,
					  metrics=[dice_coefficient])

		K.set_value(model.optimizer.lr, learningRate)
		print("[INFO] new learning rate: {}".format(
			K.get_value(model.optimizer.lr)))

	def apply_pruning_to_dense(layer):
		if isinstance(layer, tf.keras.layers.Conv2D):
			return tfmot.sparsity.keras.prune_low_magnitude(layer)
		return layer

	model_for_pruning = tf.keras.models.clone_model(
		model,
		clone_function=apply_pruning_to_dense,
	)

	model_for_pruning.summary()


	# build the path to the training plot and training history
	plotPath = "./kerasmodels/plots/resnet_fashion_mnist.png"
	jsonPath = "./kerasmodels/plots/resnet_fashion_mnist.json"

	log_dir = tempfile.mkdtemp()
	# construct the set of callbacks
	callbacks = [
		EpochCheckpoint(checkPointPath, every=saveModelEvery,
			startAt=startEpochVal),
		TrainingMonitor(plotPath,
			jsonPath=jsonPath,
			startAt=startEpochVal),
		tfmot.sparsity.keras.UpdatePruningStep(),
		# Log sparsity and other metrics in Tensorboard.
		tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)
		]


	input_ds = get_training_dataset(params, audio_adapter, audio_path )
	test_ds = get_validation_dataset(params, audio_adapter, audio_path)

	model_for_pruning.fit_generator(
		input_ds,
		validation_data=test_ds,
		steps_per_epoch=64,
		epochs=noOfEpochs,
		callbacks=callbacks,
		verbose=1)

	print(100)

	return model_for_pruning

def trainQATModelOverEpochs(noOfEpochs=20, saveModelEvery=5, startEpochVal=0, modelPath=None, learningRate=0):

	global opt
	if(learningRate==0):
		learningRate = INIT_LR
	# if there is no specific model checkpoint supplied, then initialize
	# the network and compile the model
	if modelPath is None:
		print("[INFO] compiling model...")
		model = getUnetModel("vocals")
		model.compile(loss=custom_loss, optimizer=opt,
			metrics=[dice_coefficient])
	# otherwise, we're using a checkpoint model
	else:
		# load the checkpoint from disk
		print("[INFO] loading {}...".format(modelPath))
		model = load_model(modelPath, custom_objects={"loss":custom_loss, "metrics":dice_coefficient}, compile=False)
		model.compile(loss=custom_loss, optimizer=opt,
					  metrics=[dice_coefficient])

		K.set_value(model.optimizer.lr, learningRate)
		print("[INFO] new learning rate: {}".format(
			K.get_value(model.optimizer.lr)))

	def apply_quantization_to_dense(layer):
		if isinstance(layer, tf.keras.layers.Conv2D):
			return tfmot.quantization.keras.quantize_annotate_layer(layer)
		return layer

	annotated_model = tf.keras.models.clone_model(
		model,
		clone_function=apply_quantization_to_dense,
	)

	quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
	quant_aware_model.summary()

	quant_aware_model.compile(loss=custom_loss, optimizer=opt,
					  metrics=[dice_coefficient])


	# build the path to the training plot and training history
	plotPath = "./kerasmodels/plots/resnet_fashion_mnist.png"
	jsonPath = "./kerasmodels/plots/resnet_fashion_mnist.json"

	log_dir = tempfile.mkdtemp()
	# construct the set of callbacks
	callbacks = [
		EpochCheckpoint(checkPointPath, every=saveModelEvery,
			startAt=startEpochVal),
		TrainingMonitor(plotPath,
			jsonPath=jsonPath,
			startAt=startEpochVal)]


	input_ds = get_training_dataset(params, audio_adapter, audio_path )
	test_ds = get_validation_dataset(params, audio_adapter, audio_path)

	quant_aware_model.fit_generator(
		input_ds,
		validation_data=test_ds,
		steps_per_epoch=64,
		epochs=noOfEpochs,
		callbacks=callbacks,
		verbose=1)

	print(100)

	return quant_aware_model
