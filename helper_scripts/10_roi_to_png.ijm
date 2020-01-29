function makeMask(file_image, file_roi, file_out) {
	
	// Open image and ROI Manager
	if (isOpen("ROI Manager")) {
	     selectWindow("ROI Manager");
	     run("Close");
	}

	open(file_image);
	open(file_roi);
	roiManager("Open", file_roi);

	// Make masks and save
	roiManager("Fill");
	roiManager("XOR");
	n = roiManager("count");
	for (i=0; i<n; i++){
		roiManager("Select", i);
		run("Create Mask");
	  	path = file_out + i + ".png ";
		saveAs("PNG", path);
		close();
	}

	// Close image and ROI manager
	roiManager("Delete");
	close();

	return;
}

// Get files
root = "/Users/beichenberger/Downloads/Labeling/Nuclei_other/labeled/";
list = getFileList(root);
list = Array.sort(list);

// Loop over files
for (i = 0; i<list.length; i++){
		curr = substring(list[i], 0, (lengthOf(list[i])-1));
		file_image = root + list[i] + "images/" + curr + ".tif";
		file_roi = root + list[i] + "RoiSet.zip";
		file_out = root + list[i] + "masks/mask_";
		makeMask(file_image, file_roi, file_out);
}
 