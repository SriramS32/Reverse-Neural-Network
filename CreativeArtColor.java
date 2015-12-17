/*
@author Sriram Somasundaram
 */
import java.io.*;
import java.util.*;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;

//The major methods are changeWeights(), train(), trainNodes(), changeNodes(). The main method takes up a lot of space due to processing of RGB image files.

//FOR CHANGING INITIAL SETTINGS - change Input, Didden, Output, cases static final parameters, wanted output (in setOut()), and image sizes (in main function)

public class CreativeArtColor
{
    private double[][] ktrainingInputs;
    private double[][] jnodes;
    private double[][] itargetOutput;
    private double[][] irealOutput;
    private double initError= .2;
    private double[][] iWeights;
    private double[][] hWeights;
    public static final int cases = 4;
    public static final int Input = 270000;
    public static final int Hidden = 15;
    public static final int Output = 1;
    public static final double errorMax = .000001;
    public static final int baseError = 20;
    public static final int maxCycles = 800; 
    public static final double lowerB = -1.0;
    public static final double middleB = 0.0;
    public static final double upperB = 1.0;
    public static final double upperPB = 255.0;
    public static final double learningRate = 0.25;

    /*
    Purpose: 	1. Initializes weights through randomization until the error is below a threshold initError
    			2. Trains network until error is below errorMax
    */
    public void changeWeights()
    {
        double[][] tempNodes;
        //loop over all the inputs (how many cases or pictures and such)
        double currError =1;
        int currCycles = 0;
        //this does the randomization unless we get weights with errors resulting less than the start max
        while(currError>initError && currCycles<maxCycles)
        {
            initializeWeights();
            propagateNet(cases);
            currError=error();
            currCycles++;
        }
        System.out.println("Initial Error: " + currError);
        currCycles = 0;
        while(currError>errorMax && currCycles<maxCycles)
        {
            propagateNet(cases);
            train();
            currError=error();
            if(currCycles%20==0) System.out.println(currError);
            currCycles++;
        }
        System.out.println("Final Error: " + currError);
        return;
    }

    /*
    Purpose: 	1. Iterates over all of the image cases
    			2. Implements backpropagation by changing weights for deeper hidden layer weights first and saving terms for earlier hidden layer weights
    			3. In doing so, trains weights
    */
    public void train()
    {
        for(int n=0;n<cases;n++)
        {
            //double[] thetai=new double[Output];
            double[] omegai=new double[Output];
            double[] psii = new double[Output];
            double[] bigOmega = new double[Hidden];
            for (int j=0;j<Hidden;j++) {
                for (int i=0;i<Output;i++) {
                    omegai[i]= itargetOutput[n][i]-irealOutput[n][i];
                    psii[i]=omegai[i]*dervActivation(irealOutput[n][i]);
                    hWeights[i][j]+=learningRate*psii[i]*jnodes[n][j];
                    bigOmega[j]+=psii[i]*hWeights[i][j];
                }

            }
            for (int k=0;k<Input;k++) {
                for (int j=0;j<Hidden;j++) {
                    iWeights[j][k]+=learningRate*bigOmega[j]*ktrainingInputs[n][k]*dervActivation(jnodes[n][j]);
                }
            }

        }
        return;
    }

    /*
    Purpose: 	1. Initializes input nodes through randomization until the error is below a threshold initError
    			2. Trains network until error is below errorMax
    */
    public void changeNodes()
    //only diff is calling the diff train method, diff initialize method
    {
        //double[][] tempNodes;
        //loop over all the inputs (how many cases or pictures and such)
        double currError =5;
        int currCycles = 0;
        //this does the randomization unless we get weights with errors resulting less than the start max

        while(currError>initError && currCycles<maxCycles)
        {
            initializeNodes();
            propagateNet(cases);
            currError=error();
            //System.out.println(currError);
            currCycles++;
            //printStuff(1000);
        }
        System.out.println("Initial Error: " + currError);
        currCycles = 0;
        while(currError>errorMax && currCycles<maxCycles)
        {
            propagateNet(cases);
            trainNodes();
            currError=error();
            if(currCycles%20==0) 
            {
                System.out.println(currError);
                //printStuff(currCycles);
            }
            currCycles++;
        }
        System.out.println("Final Error: " + currError);
        return;
    }

    /*
    Purpose: 	1. Iterates over all of the image cases
    			2. Implements backpropagation by changing node values for deeper hidden layer nodes first and saving terms for earlier nodes
    			3. In doing so, trains nodes. It changes the inputs through training based on desired outputs and its current weight set.
    			Note: Very similar algorithm to training weights, but certain terms have been changed based on the partial derivative with respect to input calculations of minimization of error
    */
    public void trainNodes()
    {
        for(int n=0;n<cases;n++)
        {
            //double[] thetai=new double[Output];
            double[] omegai=new double[Output];
            double[] psii = new double[Output];
            double[] bigOmega = new double[Hidden];
            for (int j=0;j<Hidden;j++) {
                for (int i=0;i<Output;i++) {
                    omegai[i]= itargetOutput[n][i]-irealOutput[n][i];
                    psii[i]=omegai[i]*dervActivation(irealOutput[n][i]);
                    jnodes[n][j] +=learningRate*psii[i]*hWeights[i][j];
                    //may need to take the hidden calculation out, depending
                    bigOmega[j]+=psii[i]*hWeights[i][j];
                }

            }
            for (int k=0;k<Input;k++) {
                for (int j=0;j<Hidden;j++) {
                    ktrainingInputs[n][k]+=learningRate*bigOmega[j]*iWeights[j][k]*dervActivation(jnodes[n][j]);
                }
            }
        }
        return;
    }

    //Purpose: to set the desired outputs based on the image set in the folder that this file is located in
    public void setOut()
    {
        itargetOutput = new double[][] {{1.0},{1.0},{0.0},{0.0}};
        //Make sure this has new outputs updated
    }

    /*
    Purpose: 	1. "train" - Creates a new neural network and trains a set of weights and prints to a file
    			2. "run" - Uses a previously created weight set in a file to create a network and propagates image inputs through that network
    			3. "reverse" - Uses a previously created weight set in a file and trains image inputs based on the constant weights, creating image files
    */
    public static void main(String[] args) throws IOException{
        Scanner in = new Scanner(System.in);
        File file = new File("weightsColor.txt");
        if (!file.exists()) file.createNewFile();
        Scanner scan = new Scanner(file);
        System.out.println("Type train or run or reverse");
        String t = in.next();
        double[][] iWeights1 = new double[Hidden][Input];
        double[][] hWeights1 = new double[Output][Hidden];
        if(t.equals("train"))
        {
            BufferedWriter bout = new BufferedWriter(new FileWriter(file));
            CreativeArtColor autobot = new CreativeArtColor(iWeights1, hWeights1);
            autobot.initializeWeights();
            //Made a buffered writer and an object of this class and initialized weights randomly

            double[][] ktrainingInputs1 = new double[cases][Input];
            //initializes a double array for inputs and then puts a combination of rgb value of picture into inputs
            BufferedImage[] images = new BufferedImage[cases];
            //array of all images, decided by the input cases earlier
            for(int n =0;n<cases;n++)
            {
                images[n] = null;
                File f = new File("image" + n + ".jpg");
                images[n] = ImageIO.read(f);
                int width = images[n].getWidth();
                int height = images[n].getHeight();
                //if you want to get less inputs than what is there, do this
                //width = 100;
                //height = 100;
                for(int i =0; i<height; i++)
                {
                    int jCount = -1;
                    for(int j = 0;j<width*3; j++)
                    {
                        jCount++;
                        int p = images[n].getRGB(jCount, i);

                        int r = (p>>16)&0xff;
                        int g = (p>>8)&0xff;
                        int b = p & 0xff;
                        //int sum = (int)Math.round((r*0.3 + g*0.589 + b*0.11)) //for grayscale;
                        //CONSIDER PUTTING IN THE 255 inverter
                        ktrainingInputs1[n][j+i*(width*3)]=r;
                        j++;
                        ktrainingInputs1[n][j+i*(width*3)]=g;
                        j++;
                        ktrainingInputs1[n][j+i*(width*3)]=b;
                        //Change based on what the desired input is
                    }

                }
            }

            autobot.setInput(ktrainingInputs1);
            //inputs are passed to the object of this class
            autobot.initOut(cases);
            autobot.initHid(cases);
            autobot.setOut();
            //setting ideal output and initializing real output and hidden layer

            autobot.changeWeights();
            //changeWeights() calls the main training algorith -> training the weights to then print out the weights to file
            iWeights1=autobot.getIWeights();
            hWeights1=autobot.getHWeights();
            for (int j=0;j<Hidden;j++)
            {
                for (int k=0;k<Input;k++)
                {
                    bout.write(iWeights1[j][k] + "\n");
                }
            }

            for(int i=0;i<Output;i++)
            {
                for (int j=0;j<Hidden;j++)
                {
                    bout.write(hWeights1[i][j] + "\n");
                }
            }
            bout.close();
            System.out.println("Trained...");
        }
        else if (t.equals("run"))
        {
            //getting the weights from the optimized weights file

            for (int j=0;j<Hidden;j++)
            {
                for (int k=0;k<Input;k++)
                {
                    iWeights1[j][k]=scan.nextDouble();
                }
            }

            for (int i=0;i<Output;i++)
            {
                for (int j=0;j<Hidden;j++)
                {
                    hWeights1[i][j]=scan.nextDouble();
                }
            }

            //getting in all the input weights that will be propagated to check output
            CreativeArtColor autobot = new CreativeArtColor(iWeights1,hWeights1);
            //need to set cases first, because of constructor
            double[][] ktrainingInputs1 = new double[cases][Input];
            autobot.initOut(cases);
            autobot.initHid(cases);
            //initializing hidden and output arrays
            BufferedImage[] images = new BufferedImage[cases];
            for(int n =0;n<cases;n++)
            {
                images[n] = null;
                File f = new File("image" + n + ".jpg");
                images[n] = ImageIO.read(f);

                int width = images[n].getWidth();
                //System.out.println(width);
                int height = images[n].getHeight();
                //System.out.println(height);

                //if you want to get less inputs than what is there, do this
                //width = 100;
                //height = 100;

                for(int i =0; i<height; i++)
                {
                    int jCount=-1;
                    for(int j = 0;j<width*3; j++)
                    {    
                        jCount++;
                        int p = images[n].getRGB(jCount, i);

                        int r = (p>>16)&0xff;
                        int g = (p>>8)&0xff;
                        int b = p & 0xff;

                        ktrainingInputs1[n][j+i*(width*3)]=r;
                        j++;
                        ktrainingInputs1[n][j+i*(width*3)]=g;
                        j++;
                        ktrainingInputs1[n][j+i*(width*3)]=b;
                    }

                }
            }

            autobot.setInput(ktrainingInputs1);
            //setting ideal output

            autobot.propagateNet(cases);
            double[][] irealOutput1 = autobot.getOutput();
            for (int n =0;n<cases;n++) {
                for (int i=0;i<Output;i++) {
                    System.out.println("Case " + n + " Output " + i + ": " + irealOutput1[n][i]);
                }
            }
            System.out.println("Done");
        }
        else if(t.equals("reverse"))
        {
            for (int j=0;j<Hidden;j++)
            {
                for (int k=0;k<Input;k++)
                {
                    iWeights1[j][k]=scan.nextDouble();
                }
            }

            for (int i=0;i<Output;i++)
            {
                for (int j=0;j<Hidden;j++)
                {
                    hWeights1[i][j]=scan.nextDouble();
                }
            }

            CreativeArtColor autobot = new CreativeArtColor(iWeights1,hWeights1);

            //taking in the inputs
            double[][] ktrainingInputs1 = new double[cases][Input];
            //initialized and then set the input nodes
            autobot.setInput(ktrainingInputs1);
            autobot.setOut();
            //setting ideal output and initializing real output
            autobot.initOut(cases);
            autobot.initHid(cases);
            //initialize output and hidden nodes

            autobot.changeNodes();

            double[][] ktrainingInputs2 = autobot.getInputs();

            int height = 300;
            int width = 300;
            //int bWidth = 100*3;
            BufferedImage[] images = new BufferedImage[cases];
            //Set to be as what the image is, can later change to be fed in.

            for(int n=0;n<cases;n++)
            {
                File f = new File("image" + n + ".jpg");

                //images[n] = new BufferedImage(width,height,3);
                //argb style, check later for third type into buffered image
                images[n]= ImageIO.read(f);
                //this initializes the image as the original image with same width height and file type
                for(int i =0; i<height; i++)
                {
                    //for(int j = 0;j<bWidth; j+=3)
                    int jCount = -1;
                    for(int j = 0;j<width*3; j++)
                    {
                        /*
                        int sum = (int)Math.round(ktrainingInputs2[n][j+i*width]);
                        if(sum>255)
                        {
                        sum = 255;
                        }
                        else if(sum <0)
                        {
                        sum = 0;
                        }
                         */

                        int r = (int)Math.round(ktrainingInputs2[n][j+i*(width*3)]);
                        j++;
                        int g = (int)Math.round(ktrainingInputs2[n][j+i*(width*3)]);
                        j++;
                        int b = (int)Math.round(ktrainingInputs2[n][j+i*(width*3)]);
                        if(r>255)
                        {
                            r = 255;
                        }
                        else if(r <0)
                        {
                            r = 0;
                        }
                        if(g>255)
                        {
                            g = 255;
                        }
                        else if(g <0)
                        {
                            g = 0;
                        }
                        if(b>255)
                        {
                            b = 255;
                        }
                        else if(b <0)
                        {
                            b = 0;
                        }

                        //after second success the if statement split into two 
                        //and was fixed to actually change the value of the set rgb.
                        //images[n].setRGB(j,i,(int)Math.round(ktrainingInputs2[n][j+i*width]));

                        //int p = (0<<24) | (r<<16) | (g<<8) | b;
                        //int p = (0<<24) | ((int)(sum*.2126)<<16) | ((int)(sum*.7152)<<8) | (int)(sum*.0722);

                        //int p = (255<<24) | (sum<<16) | (sum<<8) | sum;
                        int p = (255<<24) | (r<<16) | (g<<8) | b;
                        jCount++;

                        images[n].setRGB(jCount,i,p);
                        //images[n].setRGB(j,i,p);
                    }
                }
                File f1 = new File("image" + n + "Output.jpg");
                ImageIO.write(images[n], "gif", f1);
            }
            //bout.close();

            System.out.println("Done");
        }
    }

    public void printWeightMap(int cases, int width, int height)
    //FIX WEIGHT MAP IT NEEDS TO BE FOR EACH HIDDEN NODE, NOT PICTURE
    //FIX THe DIFFERENT STABILIZATION, DONT USE THE ACTIVATION FUNCTION, ADD BY 150, limit cap.
    {
        for (int n = 0; n<cases; n++)
        {   
            BufferedImage[] images = new BufferedImage[Hidden];
            File f = new File("image" + n + ".jpg");
            try{
                images[n] = ImageIO.read(f);}
            catch(IOException e)
            {
            }

            //Summing all the weight values from the hidden layer to each input pixel. For how the important the input pixels are initially
            for (int j = 0; j < Hidden; j++)
            {
                int[] iSum = new int[Input];
                for (int k = 0; k < Input; k=k+3)
                {
                    iSum[k] = (int)Math.round((255*activation(iWeights[j][k])));
                    iSum[k+1] = (int)Math.round((255*activation(iWeights[j][k+1])));
                    iSum[k+2] = (int)Math.round((255*activation(iWeights[j][k+2])));
                    int p = (255<<24) | (iSum[k]<<16) | (iSum[k+1]<<8) | iSum[k+2];
                    System.out.println(iSum[k] + " " + iSum[k+1] + " " + iSum[k+2]);
                    int currentH = (int)((k/3)/height);
                    int currentL = (k/3)%width;
                    images[n].setRGB(currentL, currentH, p);
                }
            }
            File f1 = new File("image" + n + "WeightOutput.jpg");
            try{ImageIO.write(images[n],"gif", f1);}
            catch(IOException e){}
        }
        System.out.println("Done");
    }

    public void initOut(int c)
    {
        irealOutput = new double[c][Output];
    }

    public void initHid(int c)
    {
        jnodes = new double[c][Hidden];
    }

    public double[][] getOutput()
    {
        return irealOutput;
    }

    public void setInput(double[][] a)
    {
        ktrainingInputs = a;
    }

    public CreativeArtColor(double[][] iWeights1, double[][] hWeights1)
    {
        iWeights=iWeights1;
        hWeights=hWeights1;
        jnodes = new double[cases][Hidden];
    }

    //Meant to be able to print a video of neural network drawing images, Currently not used
    public void printStuff(int count)
    {
        int n=0;
        File f = new File("image0.jpg");

        //images[n] = new BufferedImage(width,height,3);
        //argb style, check later for third type into buffered image
        //images[n]= ImageIO.read(f);
        BufferedImage img = null;
        try{
            img = ImageIO.read(f);
        }
        catch(IOException e){
            System.out.println(e);
        }
        //this initializes the image as the original image with same width height and file type
        int height = img.getHeight();
        //System.out.println(height);
        int width = img.getWidth();
        //System.out.println(width);
        for(int i =0; i<height; i++)
        {
            //for(int j = 0;j<bWidth; j+=3)
            for(int j = 0;j<width; j++)
            {

                int sum = (int)Math.round(ktrainingInputs[n][j+i*width]);
                if(sum>255)
                {
                    sum = 255;
                }
                else if(sum <0)
                {
                    sum = 0;
                }

                /*
                int r = (int)Math.round(ktrainingInputs2[n][j+i*width]);
                int g = (int)Math.round(ktrainingInputs2[n][j+i*width+1]);
                int b = (int)Math.round(ktrainingInputs2[n][j+i*width+2]);
                if(r>255)
                {
                r = 255;
                }
                else if(r <0)
                {
                r = 0;
                }
                if(g>255)
                {
                g = 255;
                }
                else if(g <0)
                {
                g = 0;
                }
                if(b>255)
                {
                b = 255;
                }
                else if(b <0)
                {
                b = 0;
                }
                 */
                //after second success the if statement split into two 
                //and was fixed to actually change the value of the set rgb.
                //images[n].setRGB(j,i,(int)Math.round(ktrainingInputs2[n][j+i*width]));

                //int p = (0<<24) | (r<<16) | (g<<8) | b;
                //int p = (0<<24) | ((int)(sum*.2126)<<16) | ((int)(sum*.7152)<<8) | (int)(sum*.0722);
                int p = (255<<24) | (sum<<16) | (sum<<8) | sum;
                //images[n].setRGB((int)(j/3),i,p);
                img.setRGB(j,i,p);
            }
        }
        File f1 = new File("VideoReal" + count + "Output.jpg");
        try{
            ImageIO.write(img, "gif", f1);
        }
        catch(IOException e){
            System.out.println(e);
        }
    }

    public double randBound (double lowerB1, double upperB1)
    {
        return Math.random()*(upperB1-lowerB1)+lowerB1;
    }

    public void initializeWeights()
    {
        for(int j=0;j<Hidden;j++)
        {

            for(int k=0;k<Input;k++)
            {

                iWeights[j][k]=randBound(lowerB,upperB);
                //System.out.println("weights[i][j][k]: " + weights[i][j][k]);
            }
        }

        for(int i=0;i<Output;i++)
        {
            for(int j=0;j<Hidden;j++)
            {
                hWeights[i][j]=randBound(lowerB,upperB);
            }
        }
        return;
    }

    public void initializeNodes()
    {
        for(int j=0;j<cases;j++)
        {
            for(int k=0;k<Input;k++)
            {
                ktrainingInputs[j][k]=randBound(middleB,upperPB);
                //System.out.println("ktrainingInputs[j][k]: " + ktrainingInputs[j][k]);
            }
        }
        //This part is if you want to initialize hidden nodes as well for implications initial learning calculations
        /*    
        for(int i=0;i<cases;i++)
        {
        for(int j=0;j<Hidden;j++)
        {
        jnodes[cases][j]=randBound(lowerB,upperB);
        //System.out.println("jnodes[j][k]: " + jnodes[j][k]);
        }
        }
         */
        return;
    }

    public double[][] getIWeights()
    {
        return iWeights;
    }

    public double[][] getHWeights()
    {
        return hWeights;
    }

    public double[][] getInputs()
    {
        return ktrainingInputs;
    }

    public double activation(double x)
    {
        return (1/(1+Math.exp(-x)));
    }

    public double derv(double x)
    {
        return x*(1-x);
    }

    public double dervActivation(double x)
    {
        return derv(activation(x));
    }

    public void calculateHidden(double[] input, int n)
    {
        double[] output = new double[Hidden];
        //double temp=0; For some weird mistake, the temp variable was reset on the outside here initially
        //but I changed it to be reset for each iteration of the outer loop just how calculateOutput was at
        for (int j=0;j<Hidden;j++)
        {
            double temp=0; 
            for(int k=0;k<input.length;k++)
            {
                temp += input[k]*iWeights[j][k];
            }
            output[j]=activation(temp);
        }
        jnodes[n] = output;
        return;
    }

    public void calculateOutput(double[] input, int n)
    {
        /*
        So lets just say uh this calculates like the hidden layer,
        you are given a bunch of inputs (50 by 50 pic, 2500 inputs) and then you have
        weights(where layer 0 is the "first output") Iterate over the inputs first and then the hidden nodes.
         */
        double[] output = new double[Output];
        for (int i=0;i<Output;i++)
        {
            double temp=0;
            for(int j=0;j<input.length;j++)
            {
                temp += input[j]*hWeights[i][j];
            }
            output[i]=activation(temp);
            //System.out.println(output[i]);
        }
        irealOutput[n] = output;
        return;
    }

    public double error()
    {
        /*
        set an error 0 and then sum equals consistent squares of diff between target and calculated,
        uses a temp variable to hold the evaluated inputs. Correct indices.
         */
        double error = 0.0;
        for(int n=0;n<cases;n++)
        {
            for (int i=0;i<Output;i++) {
                error+=Math.pow((itargetOutput[n][i]-irealOutput[n][i]),2);
            }
        }
        //System.out.println(error);
        return (error/2.0);
    }

    public void propagateNet(int a)
    {
        for (int n =0;n<a;n++) {
            calculateHidden(ktrainingInputs[n],n);
            calculateOutput(jnodes[n],n);
        }
        return;
    }
}