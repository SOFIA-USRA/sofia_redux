The bad pixel mask should be a single-extension FITS file containing integer
values, where 1 indicates a good pixel, 0 indicates a bad pixel, and
2 indicates a reference pixel.

Note that the default files are included in the source distribution of this
package, but not in the pip or conda versions. They may be downloaded
separately, if desired, from the
`GitHub repository <https://github.com/SOFIA-USRA/sofia_redux>`__.
Otherwise, the software will attempt to automatically download and
cache the reference file as needed.
