Getting started
===============

This step-by-step guide is intended to assist in manually reducing EXES
data. A more general introduction to the Redux GUI and reduction algorithms
can be found in the EXES Redux User's Manual.

Cross-dispersed data
====================

Overall process: get the order spacing right, get the slit rotation
right, get the extraction right, get the central wavelength right, get
the dispersion right.

1.  Load in the data (File -> Open New Reduction).

    -  If there is more than one science file, you may want to select
       just one to start with, along with the flat, until you are sure
       you have the correct distortion parameters (step 2-3).

    -  If RAW files are available, load those instead of the truly raw
       files. If not, you may want to generate them for all science
       files first (by running Load Data), then start again with the RAW
       files.

2.  Step to: Make Flat.

    -  Verify that the step completed successfully (no ERROR messages in
       the terminal).

    -  Verify the spacing between orders.

       -  Measure the actual spacing in the image (use *d* in ximgtool
          and right-click to draw a horizontal line), and verify that it
          roughly matches the predicted spacing (printed to the
          terminal).

       -  Verify that the calculated spacing matches the predicted order
          spacing.

    -  Verify that the orders look roughly rectilinear.

    -  If any of these checks fail: Undo the Make Flat step (or Reset).

       -  If the predicted spacing seems off, it can be fixed by
          changing XDFL. Edit Param for the Load Data step to enter a
          new value. The value used in the first run should have been
          printed to the terminal. **Increasing XDFL will increase the
          predicted spacing.**

          -  The predicted spacing does not have to match the actual
             spacing exactly. It just needs to be close enough for the
             optimization algorithm to find the right value.

          -  Typical values for XDFL are about 85-110. If you find you
             have to use values far outside that range, something else
             is probably wrong.

       -  If the calculated spacing or the rotation of the orders seems
          off, it is likely that the edge detection routine failed. Edit
          Param for the Make Flat step, select Debug, and run again.
          Debugging will stop the reduction with an error message and
          display some useful images. Buffer 1 is the undistorted flat,
          Buffer 2 is the edge-enhanced image, Buffer 3 is the FFT of
          that image, Buffer 4 is the region of the FFT near where a
          peak is predicted (in the first and second harmonics of the
          FFT), and Buffer 5 is the calculated illumination mask. If the
          edge detection is working correctly, there will be a clear
          peak in at least one of the stamps in Buffer 4, and the
          illumination mask in Buffer 5 will match the undistorted flat.
          If this is not the case, try a different Edge enhancement
          method in the Make Flat parameters and run again. Which method
          works best sometimes depends on the illumination
          characteristics of the particular flat being reduced.

       -  If you can't get the edge detection algorithm to work at all,
          it may be that the flat just doesn't have obvious enough
          edges. You may be able to get close to a good correction by
          deselecting Optimize rotation angle, and editing the Starting
          rotation angle and/or Predicted spacing by hand.

       -  If the order spacing and rotation look okay, but the pipeline
          reports that it couldn't find the order edges, try increasing
          the Threshold factor to make the inter-order spacing a little
          wider.

       -  When you have a reasonable undistorted flat, turn off Debug
          and run the step once more.

3.  Step to: Undistort.

    -  Check that the distortion correction looks okay, and there are no
       bad nod pairs. Use Display -> Quick Look to step through each input
       file. Use Buffer -> Cycle Frames to step through each nod pair in
       the file.

       -  If emission or absorption features appear tilted across the
          orders, you may need to modify the slit rotation. Reset the
          reduction and Edit Param for Load Data to change the slit
          rotation. **Increasing the slit rotation angle rotates the
          lines clockwise.**

          -  To get the slit rotation exactly right, it may be helpful
             to extract sky spectra at the top and bottom of the order
             and compare features in the extracted apertures. To do
             this, Edit Param for Subtract Nods to skip nod subtraction,
             and for Find Objects and Set Apertures to set manual
             apertures, then Step to: Extract Spectra. Reset, edit Slit
             Rotation, and re-reduce as necessary.

       -  If there are bad nod pairs (eg. if the observation was
          aborted), note the file number and frame number. Edit Param
          for Coadd Pairs and enter any frames that need to be excluded
          in Exclude Pairs. Files are separated by semi-colons, frames
          are separated by commas. For example, to exclude the first
          frame from the first file, and the last frame from last file
          in a 3-file, 6-nod sequence, enter 1;;6.

4.  Step to: Make Profiles. Step to: Set Apertures.

    -  Check that the apertures have been correctly set.

       -  Use Display -> Quick Look to check each image; look for light
          blue lines over the center of each aperture in the image. Look
          for a green overlay over most of the source in the spatial
          profile. If the apertures are badly placed, Undo.

       -  If the objects look misidentified, set them by hand by using
          Fix or Guess parameters in Find Objects.

       -  If the apertures look wrong, set them by hand using the
          Override parameters in Set Apertures.

5.  Step to: Extract Spectra. Verify the spectra from all
    apertures/files look about the same. If not, some non-standard
    extraction parameters may need to be used (eg. turn off bad pixel
    correction, use median spatial profile, set background regions,
    etc.).

    -  A single file is usually loaded into xvspec. More can be added in
       for direct comparison using Add Spectra. They can also be
       viewed/combined in the separate tool called xcombspec.

6.  Step to: Merge Orders. Click Cancel in the Refine Wavecal GUI.
    Compare the atmospheric transmission in the Transmission pane in
    xvspec to the merged spectrum to identify an atmospheric feature. It
    can be overplotted on the data with Plot -> Overplot -> Transmission.

    -  The wavenumber of the feature can be measured by selecting
       Transmission, clicking on the plot, typing *f* over the new
       window, and clicking on the left, then the right, side of the
       feature. The centroid of the feature will be reported in the bar
       at the top of the window.

    -  Measure the wavenumber of the corresponding feature in the
       science spectrum by clicking Flux and repeating the above process
       to centroid the line. The initial central wavenumber is sometimes
       very far off, which can make identification of a corresponding
       feature difficult.

7.  Undo. Step to: Refine Wavecal. Identify the feature in the spectrum
    (click on the order, centroid the line, click Add from Plot in the
    little GUI window), then edit the wavelength to the measured
    wavelength from the Transmission spectrum. Click Done.

8.  Click Reset. The calculated wavenumber will now be used to update
    the distortion/dispersion parameters. Step to: Refine Wavecal to
    redo the full reduction.

    -  Verify that the distortion correction still looks right.

    -  Verify that the identified feature now has the correct
       wavenumber. If the central wavelength was far off, you may have
       to Refine it, and repeat the reduction again.

9.  If there are overlapping orders, the dispersion may need to be tuned
    to get the lines to appear at the same wavelength in each order.

    -  The separate tool xmergeorders can be used to check the
       overlapping regions.

    -  For cross-dispersed spectra, the dispersion is tuned with the
       HRFL parameter. **Reducing HRFL decreases the wavenumber of the
       left end of the spectrum** (eg. if a feature at the left end of
       Order 2 is to the right of a feature at the right end of Order 1,
       decreasing HRFL slightly may fix it). To edit HRFL, Reset the
       reduction, Edit Param for Load Data, and repeat from step 7.
       Changing HRFL will change the wavelength calibration, which
       changes the distortion correction, so the wavecal will have to be
       Refined, and possibly Refined again.

       -  Typical values for HRFL are about 85-110, as for XDFL.

10. Step to: Merge Orders.

    -  For spectra with overlap regions, verify that the final spectrum
       looks right, eg. that there are not discontinuous jumps, or
       excessively noisy overlap regions. If there are, Undo and Edit
       Param for Merge Orders. Try higher or lower values for the
       Selection threshold. Spectra can also be merged by hand using the
       xmergeorders tool.

11. Finally: save the reduction parameters for the observation, using
    Parameters -> Save Current Parameters. These parameters serve as a
    record of the manual reduction, and can later be directly loaded
    into either the GUI or the automatic pipeline to repeat the same
    reduction.

Long-slit data
==============

Overall process: Same as for cross-dispersed data, except that there is
no order spacing to worry about in the distortion correction, and no
order merging in the final product.

1. Load in the data (File -> Open New Reduction).

2. Step to: Undistort.

   -  Check that the distortion correction looks okay, and there are no
      bad nod pairs. Use Display -> Quick Look to step through each input
      file. Use Buffer -> Cycle Frames to step through each nod pair in the
      file.

      -  If emission or absorption features appear tilted across the
         order, you may need to modify the slit rotation. Reset the
         reduction and Edit Param for Load Data to change the slit
         rotation. **Decreasing the slit rotation angle rotates the
         lines clockwise.**

         -  To get the slit rotation exactly right, it may be helpful to
            extract sky apertures at the top and bottom of the order and
            compare features in the extracted apertures. To do this,
            Edit Param for Subtract Nods to skip nod subtraction, and
            for Find Objects and Set Apertures to set manual apertures,
            then Step to: Extract Spectra. Reset, edit Slit Rotation,
            and re-reduce as necessary.

      -  If there are bad nod pairs (eg. if the observation was
         aborted), note the file number and frame number. Edit Param for
         Coadd Pairs and enter any frames that need to be excluded in
         Exclude Pairs. Files are separated by semi-colons, frames are
         separated by commas. For example, to exclude the first frame
         from the first file, and the last frame from last file in a
         3-file, 6-nod sequence, enter 1;;6.

3. Step to: Make Profiles. Step to: Set Apertures.

   -  Check that the apertures have been correctly set.

      -  Use Display -> Quick Look to check each image; look for light blue
         lines over the center of each aperture in the image. Look for a
         green overlay over most of the source in the spatial profile.
         If the apertures or objects are bad, Undo.

      -  If the objects look misidentified, set them by hand by using
         Fix or Guess parameters in Find Objects.

      -  If the apertures look wrong, set them by hand using the
         Override paramters in Set Apertures.

4. Step to: Extract Spectra. Verify the spectra from all apertures/files
   look about the same. If not, some non-standard extraction parameters
   may need to be used (eg. turn off bad pixel correction, use median
   spatial profile, set background regions, etc.).

   -  A single file is usually loaded into xvspec. More can be added in
      for direct comparison using Add Spectra. They can also be
      viewed/combined in the separate tool called xcombspec.

5. Step to: Refine Wavecal. Compare the atmospheric transmission in the
   Transmission pane in xvspec to the combined spectrum to identify an
   atmospheric feature. It can be overplotted on the data with
   Plot -> Overplot -> Transmission.

   -  The wavenumber of the feature can be measured by selecting
      Transmission, clicking on the plot, typing *f* over the new
      window, and clicking on either side of the feature. The centroid
      of the feature will be reported in the top of the window.

   -  Select the corresponding feature in the science spectrum by
      clicking Flux and repeating the above process to centroid the
      line. Click Add from Plot in the little GUI window, then edit the
      wavelength to the measured wavelength from the Transmission
      spectrum. Click Done.

6. Click Reset. The calculated wavenumber will now be used to modify the
   distortion/dispersion parameters. Step to: Refine Wavecal to redo the
   full reduction.

   -  Verify that the identified feature now has the correct wavenumber.
      If the initial central wavelength was far off, you may have to
      Refine it, and repeat the reduction again.

7. The dispersion may need to be tuned to get the lines to appear at the
   correct wavelengths all the way across the spectrum.

   -  For long-slit spectra, the dispersion is tuned with the XDFL
      parameter. **Reducing XDFL decreases the wavenumber of the left
      end of the spectrum and increases the wavenumber of the right
      end,** when the central wavelength is correctly identified. To
      edit XDFL, Reset the reduction, Edit Param for Load Data, and
      repeat from step 6. Changing XDFL will change the wavelength
      calibration, which changes the distortion correction, so the
      wavecal will have to be Refined, and possibly Refined again.

8. Finally: save the reduction parameters for the observation, using
   Parameters -> Save Current Parameters. These parameters serve as a
   record of the manual reduction, and can later be directly loaded into
   either the GUI or the automatic pipeline to repeat the same
   reduction.
