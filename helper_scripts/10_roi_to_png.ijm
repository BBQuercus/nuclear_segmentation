root = "/Users/beichenberger/Downloads/Nuclei_dapi/unlabeled/";
t = "Hela-R-No-05";
roiManager("Fill");
roiManager("XOR");
for (i=0; i<=1000; i++){
	roiManager("Select", i);
	run("Create Mask");
  	path = root + t + "/masks/mask_" + i + ".png ";
	saveAs("PNG", path);
	close();
}
