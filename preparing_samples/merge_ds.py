import numpy as np
import h5py
import glob
import tqdm
import yaml
import argparse



def get_parser():
    """
    Argument parser for Merging script.

    Returns
    -------
    args: parse_args
    """
    parser = argparse.ArgumentParser(description="Merging command lineoptions.")

    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        required=True,
        help="Enter the name of the config file to merge the samples.",
    )

    return parser.parse_args()
class MergeSamples():

    def __init__(self, config):
        self.config = config
        self.samples = config['samples']
        self.output_filename = config['output_filename']
        self.total_size = int(float(config['total_size']))
        self.batch_size = int(float(config.get('batch_size', 50000)))
        self.tracks_name = config['tracks_name']
        self.cells_name = config['cells_name']
        self.n_jets_per_file = 5_000_000
        assert int(sum([val['fraction'] for val in self.samples.values()])) == 1, "Sum of the fractions is not equal to 1!"

        
        weights = {key : val['fraction'] for key, val in self.samples.items()}
        self.batch_sizes = self._split(np.arange(self.batch_size), weights)
        print(self.batch_sizes)
        self.samples_files, self.num_batches = self._prepare_samples()
        print(self.num_batches)
        self.min_num_batches = np.min(list(self.num_batches.values()))
        print(f"Max available size is {self.min_num_batches * self.batch_size} jets")

    def _prepare_samples(self):
        samples_files = {}
        num_batches = {}
        for key, value in self.samples.items():
            # files = glob.glob(value['file_path'] + '*h5')
            files = [value['file_path']]
            samples_files[key] = [self.GetBatchesPerFile(file, len(self.batch_sizes[key])) for file in files]
            num_batches[key] = num_batches.get(key, 0) + sum([len(el[1]) for el in samples_files[key]])
        return samples_files, num_batches

    def _split(self, original_list, weight_list):
        sublists = {}
        prev_index = 0
        for key, weight in weight_list.items():
            next_index = prev_index + np.ceil( (len(original_list) * weight) )
            
            sublists[key] = np.array(original_list[int(prev_index) : int(next_index)])
            prev_index = next_index
        return sublists
            
    def run(self):
        n_jets = 0
        n_curr_jets = 0
        prefix = 0
        n_batches = 0
        generators = {key: self.jets_generator(value, self.tracks_name, self.cells_name) for key, value in self.samples_files.items()}
        rng = np.random.default_rng()
        create_file = True
        pbar = tqdm.tqdm(total=self.total_size)
        displayed_writing_output = True
        while n_jets < self.total_size and n_batches < self.min_num_batches:
            
            jets = []
            tracks = []
            cells = []
            for key in generators.keys():
                gen_jets, gen_tracks, gen_cells = next(generators[key])
                jets.append(gen_jets)
                tracks.append(gen_tracks)
                if gen_cells is not None:
                    cells.append(gen_cells)
                else:
                    cells = None
            jets = np.concatenate(jets, axis=0)
            tracks = np.concatenate(tracks, axis=0)
            if cells is not None:
                cells = np.concatenate(cells, axis=0)
            indices = rng.permutation(self.batch_size)
            jets = jets[indices]
            tracks = tracks[indices]
            if cells is not None:
                cells = cells[indices]
            pbar.update(jets.size)
            if create_file:
                create_file = False  # pylint: disable=W0201:
                # write to file by creating dataset
                pbar.write(f"Creating output file: {self.output_filename}_{prefix}.h5")
                with h5py.File(self.output_filename + "_" + str(prefix) + ".h5", "w") as out_file:
                    out_file.create_dataset(
                        "jets",
                        data=jets,
                        compression="gzip",
                        chunks=True,
                        maxshape=(None,),
                    )
                    out_file.create_dataset(
                        self.tracks_name,
                        data=tracks,
                        compression="gzip",
                        chunks=True,
                        maxshape=(None, tracks.shape[1]),
                    )
                    if cells is not None:
                        out_file.create_dataset(
                            self.cells_name,
                            data=cells,
                            compression="gzip",
                            chunks=True,
                            maxshape=(None, cells.shape[1]),
                        )
            else:
                # appending to existing dataset
                if displayed_writing_output:
                    pbar.write(f"Writing to output file: {self.output_filename}_{prefix}.h5")
                with h5py.File(self.output_filename + "_" + str(prefix) + ".h5", "a") as out_file:
                    out_file["jets"].resize(
                        (out_file["jets"].shape[0] + jets.shape[0]),
                        axis=0,
                    )
                    out_file["jets"][-jets.shape[0] :] = jets
                    out_file[self.tracks_name].resize(
                        (
                            out_file[self.tracks_name].shape[0]
                            + tracks.shape[0]
                        ),
                        axis=0,
                    )
                    out_file[self.tracks_name][
                        -tracks.shape[0] :
                    ] = tracks
                    if cells is not None:
                        out_file[self.cells_name].resize(
                            (
                                out_file[self.cells_name].shape[0]
                                + cells.shape[0]
                            ),
                            axis=0,
                        )
                        out_file[self.cells_name][
                            -cells.shape[0] :
                        ] = cells
                displayed_writing_output = False
            n_jets += self.batch_size
            n_curr_jets += self.batch_size
            n_batches += 1
            if n_curr_jets >= self.n_jets_per_file:
                n_curr_jets = 0
                prefix += 1
                create_file = True

    def GetBatchesPerFile(self, filename: str, batch_size = 50000):
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
        print(filename)
        with h5py.File(filename, "r") as data_set:
            # get total number of jets in file
            total_n_jets = len(data_set["jets"])
            
            # first tuple is given by (0, batch_size)
            start_batch = 0
            end_batch = batch_size
            indices_batches = [(start_batch, end_batch)]
            # get remaining tuples of indices defining the batches
            while (end_batch + batch_size) <= total_n_jets:
                start_batch += batch_size
                end_batch = start_batch + batch_size
                indices_batches.append((start_batch, end_batch))
        return (filename, indices_batches)


    def jets_generator(self, files_in_batches: list, tracks_name = 'tracks', cells_name = 'cells'):
        """
        Helper function to extract jet, track and cell information from a h5 ntuple.

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
                    tracks = data_set[tracks_name][batch[0] : batch[1]]
                    if cells_name in data_set.keys():
                        cells = data_set[cells_name][batch[0] : batch[1]]
                    else:
                        cells = None
                    yield (jets, tracks, cells)

if __name__ == "__main__":
    args = get_parser()

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    preparation_tool = MergeSamples(config=config)
    preparation_tool.run()