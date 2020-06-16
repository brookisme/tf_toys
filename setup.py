from distutils.core import setup
setup(
  name = 'tf_toys',
  packages = ['tf_toys'],
  version = '0.0.0.1',
  description = 'tf_toys: dummy generator and model for testing workflows',
  author = 'Brookie Guzder-Williams',
  author_email = 'brook.williams@gmail.com',
  url = 'https://github.com/brookisme/tf_toys',
  download_url = 'https://github.com/brookisme/tf_toys/tarball/0.1',
  keywords = ['python','tensorflow','model','generator','dev'],
  include_package_data=True,
  data_files=[
    (
      'config',[]
    )
  ],
  classifiers = [],
  entry_points={
      'console_scripts': [
      ]
  }
)