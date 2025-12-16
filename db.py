try:
    import dataset
except ImportError:
    raise ImportError("You don't have the module `dataset` installed, which we use as an sqlite wrapper.\n"
                      "Install it with `pip install dataset==1.6.2`")

import os

class Database:

    def __init__(self, data_path, dbname="database.db"):
        os.makedirs(data_path, exist_ok=True)

        self.db = dataset.connect("sqlite:///" + os.path.join(data_path, dbname))
        print("Initialized database with existing tables:", self.db.tables)


        self.flagged_table = self.db['flagged']

    def add_flagged_image(self, id, url, file_name, species, query, filter, time):
        """Add flagged image to database

        Args:
            id (str): iNat image ID
            url (str): Amazon S3 image URL
            file_name (str): Local image filepath
            filter (Optional[str]): The species filter used to find the image
        """
        self.flagged_table.insert_ignore(dict(id=id, url=url, file_name=file_name, species=species, query=query, filter=filter, time=time), ['id'])

    def add_flagged_images(self, images):
        """Add many flagged images to database"""
        # self.flagged_table.insert_many(images)
        for image in images:
            self.add_flagged_image(**image)

    def get_flagged_images(self):
        """Get all flagged images

        Returns:
            List[Tuple]: A list of tuples [(id, inat_id, URL, file_name), ...]
        """
        images = list(self.flagged_table.all())
        return images


if __name__ == "__main__":
    db = Database("data/")

    db.add_flagged_image(3778, 'some url', 'asdf.jpg', None)

    images = db.get_flagged_images()
    print('Flagged images:')
    print(images)