# GUI feedback checklist

The user reported full success testing version 0.2.1. This checklist is retained for future releases and other machines. Automated tests cover real processing, metadata, cancellation and setup; the checks below cover actual desktop interaction and real-photo quality.

Launch the AppImage without arguments. You can also use `./photo-denoise` from the source clone. Keep your originals; outputs use a separate suffix by default.

1. **First startup and Settings:** choose CUDA on RTX 3060, or Vulkan and your chosen GPU. Check that device labels, download progress and errors are readable. Reopen the app and confirm your choice persists. Change runtime in Settings. Testing every runtime is optional; CUDA and Vulkan are the useful priorities on this machine.
2. **Desktop usability:** check text size, window resizing, file dialogs, keyboard navigation and the original/result previews on your actual display. Add a photo with EXIF orientation and confirm the preview looks upright. Try paths containing spaces and non-ASCII characters.
3. **Photo results:** process a real noisy JPEG, an 8-bit PNG and a 16-bit TIFF if available. Compare originals and results at 100% in your viewer. Try amount 30%, 80% and 100%; for DRUNet try noise levels 5, 15 and 30. Look for lost fine detail, color changes and tile boundaries. Vulkan supports only DRUNet.
4. **Batch/output behavior:** add a folder, select a separate output folder, and confirm the queue, progress, preserved subfolders and saved files make sense. Try running again with existing outputs protected, then deliberately allow replacement of outputs. Originals must stay intact.
5. **Cancellation and recovery:** cancel a batch and confirm the window remains responsive and reports what happened. Completed outputs may remain. Cancel setup if you are installing another runtime, then retry. Check that Settings still opens afterwards.
6. **CLI backup:** run the AppImage with `doctor` and process a photo using `denoise`; for Vulkan add `--model drunet`. Check that the runtime installed through the GUI is reused.

Reply with the runtime, GPU, formats tried, and either “works” or the steps that caused a problem. Include exact error text or a screenshot for UI problems. Describe denoising quality separately from app behavior; model quality on real camera noise varies. No need to upload private photos or GPS metadata.
