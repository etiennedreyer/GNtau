import pandas as pd
import numpy as np
import h5py
import os, glob
import tqdm

def GetBatchesPerFile(filename: str, batch_size = 500000):
    """
    Split the file into batches to avoid that the loaded data is too large.

    Parameters
    ----------
    filename : str
        name of file to be split in batches

    Returns
    -------
    str
        filename
    list
        tuples of start and end index of batch

    """
    with h5py.File(filename, "r") as data_set:
        # get total number of jets in file
        total_n_jets = len(data_set["jets"])
        # first tuple is given by (0, batch_size)
        start_batch = 0
        end_batch = batch_size
        indices_batches = [(start_batch, end_batch)]
        # get remaining tuples of indices defining the batches
        while end_batch <= total_n_jets:
            start_batch += batch_size
            end_batch = start_batch + batch_size
            indices_batches.append((start_batch, end_batch))
    return (filename, indices_batches)

def building_new(var, label, first_array):
    new_array = np.zeros(
        first_array.size, 
        dtype=(first_array.dtype.descr + [(label, '<f4')]))
    existing_keys = list(first_array.dtype.fields.keys())
    new_array[existing_keys] = first_array[existing_keys]
    new_array[label] = var
    return new_array

def jets_generator(files_in_batches: list, jet_type, tracks_name = 'tracks', cells_name = 'cells'):
    """
    Helper function to extract jet and track information from a h5 ntuple.

    Parameters
    ----------
    files_in_batches : list
    tuples of filename and tuple of start and end index of batch

    Yields
    -------
    numpy.ndarray
    jets
    numpy.ndarray
    tracks
    numpy.ndarray
    cells
    """
    for filename, batches in files_in_batches:
        with h5py.File(filename, "r") as data_set:
            for batch in batches:
                # load jets in batches
                jets = data_set["jets"][batch[0] : batch[1]]
                # indices_to_remove = GetJetsToRemove(jet_type, jets)
                # jets = np.delete(jets, indices_to_remove)
                tracks = data_set[tracks_name][batch[0] : batch[1]]
                cells = data_set[cells_name][batch[0] : batch[1]]
                # tracks = np.delete(tracks, indices_to_remove, axis=0)
                yield (jets, tracks, cells)


def labelDataset(jet_type, input_files, output_file, tracks_name='tracks', cells_name = 'cells', n_jets_to_get = 50000, n_jets_per_file=int(1e6)):
    jet_type_dict = {
        'tau' : 5,
        'jet' : 0
    }
    create_file = True
    jets_curr_file = 0
    _output_file = output_file
    displayed_writing_output = True
    files_in_batches = map(GetBatchesPerFile, input_files)
    pbar = tqdm.tqdm(total=n_jets_to_get)
    output_file = f'{_output_file[:-3]}_{n_jets_to_get // n_jets_per_file}.h5'
    curr_jets = 0
    for jets, tracks, cells in jets_generator(files_in_batches, jet_type):
        if len(jets) == 0:
            continue
        jets = building_new(np.ones(jets.size) * jet_type_dict[jet_type], 'HadronConeExclTruthLabelID', jets)
        jets = building_new(tracks['jetSeedPt'][:, 0], 'pt', jets)
        jets = building_new(np.abs(tracks['jetSeedEta'][:, 0]), 'absEta', jets)
        jets = building_new(np.arange(curr_jets, jets.shape[0]), 'eventNumber', jets)
        mask = ~(np.isnan(jets['pt']) | np.isnan(jets['absEta']))
        if np.sum(mask) != len(mask):
            print('Err: ', np.sum(~mask))
        jets = jets[mask]
        tracks = tracks[mask]
        cells = cells[mask]
        pbar.update(jets.size)
        n_jets_to_get -= jets.size
        if jets_curr_file >= n_jets_per_file:
             create_file = True
             jets_curr_file = 0
             output_file = f'{_output_file[:-3]}_{n_jets_to_get // n_jets_per_file}.h5'
        else:
             jets_curr_file += jets.size
        if create_file:
            pbar.write("Creating output file: " + output_file)
            create_file = False  # pylint: disable=W0201:
            # write to file by creating dataset
            with h5py.File(output_file, "w") as out_file:
                out_file.create_dataset(
                    "jets",
                    data=jets,
                    compression="gzip",
                    chunks=True,
                    maxshape=(None,),
                )
                out_file.create_dataset(
                    tracks_name,
                    data=tracks,
                    compression="gzip",
                    chunks=True,
                    maxshape=(None, tracks.shape[1]),
                )
                out_file.create_dataset(
                    cells_name,
                    data=cells,
                    compression="gzip",
                    chunks=True,
                    maxshape=(None, cells.shape[1]),
                )
        else:
            # appending to existing dataset
            if displayed_writing_output:
                pbar.write("Writing to output file: " + output_file)
            with h5py.File(output_file, "a") as out_file:
                out_file["jets"].resize(
                    (out_file["jets"].shape[0] + jets.shape[0]),
                    axis=0,
                )
                out_file["jets"][-jets.shape[0] :] = jets
                out_file[tracks_name].resize(
                    (
                        out_file[tracks_name].shape[0]
                        + tracks.shape[0]
                    ),
                    axis=0,
                )
                out_file[tracks_name][
                    -tracks.shape[0] :
                ] = tracks
                out_file[cells_name].resize(
                    (
                        out_file[cells_name].shape[0]
                        + cells.shape[0]
                    ),
                    axis=0,
                )
                out_file[cells_name][
                    -cells.shape[0] :
                ] = cells
            displayed_writing_output = False
        if n_jets_to_get <= 0:
            break
    pbar.close()
    

tau_dataset = {
    '/storage/agrp/dreyet/GNtau/samples/v04/user.ntamir.GNsamps_ntup_slim_Gammatt_MC21.GNTau_MxAOD_Gammatautau_MC21_slim.801002_v1_converted/user.ntamir.34744426._000001.ntuple.h5': 5447081,
    '/storage/agrp/dreyet/GNtau/samples/v04/user.ntamir.GNsamps_ntup_slim_Gammatt_MC21.GNTau_MxAOD_Gammatautau_MC21_slim.801002_v1_converted/user.ntamir.34744426._000002.ntuple.h5': 969371,
    '/storage/agrp/dreyet/GNtau/samples/v04/user.ntamir.GNsamps_ntup_slim_Gammatt_MC21.GNTau_MxAOD_Gammatautau_MC21_slim.801002_v1_converted/user.ntamir.34744426._000003.ntuple.h5': 4846431,
    '/storage/agrp/dreyet/GNtau/samples/v04/user.ntamir.GNsamps_ntup_slim_Gammatt_MC21.GNTau_MxAOD_Gammatautau_MC21_slim.801002_v1_converted/user.ntamir.34744426._000004.ntuple.h5': 4106447,
    '/storage/agrp/dreyet/GNtau/samples/v04/user.ntamir.GNsamps_ntup_slim_Gammatt_MC21.GNTau_MxAOD_Gammatautau_MC21_slim.801002_v1_converted/user.ntamir.34744426._000005.ntuple.h5': 969550,
}

jet_dataset = {
    'test_partial/test_XM_1.h5': 5000000,
    'test_partial/test_XM_2.h5': 5000000,
    'test_partial/test_XM_3.h5': 5000000,
    'test_partial/test_XM_4.h5': 5000000,
    'test_partial/test_XM_5.h5': 5000000,
    'test_partial/test_XM_6.h5': 5000000,
    'test_partial/test_XM_7.h5': 5000000,
    'test_partial/test_XM_8.h5': 5000000,
    'test_partial/test_XM_9.h5': 5000000,
    'test_partial/test_XM_10.h5': 5000000,
    'test_partial/test_XM_11.h5': 5000000,
    'test_partial/test_XM_12.h5': 5000000,
    'test_partial/test_XM_13.h5': 5000000,
    'test_partial/test_XM_14.h5': 5000000,
}

datasets = {
    'tau' : tau_dataset,
    'jet' : jet_dataset,
}

for samp, samp_files in datasets.items():

    for i, (samp_file, n_jets) in enumerate(samp_files.items()):
        print(f'DATASET: {samp_file}')
        labelDataset(samp, [samp_file], f'{samp}_samp{i}.h5', n_jets_to_get=n_jets, n_jets_per_file=n_jets)
