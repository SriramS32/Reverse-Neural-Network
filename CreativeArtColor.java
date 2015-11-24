/*
@author Sriram Somasundaram
 */
import java.io.*;
import java.util.*;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
//FOR CHANGING INIT SETTINGS - change input, hidden, output, cases number in main methods, and the wanted output,
//AND IMAGE SIZES in reverse method and end of train.
public class CreativeArtColor
{
    private double[][] ktrainingInputs;
    private double[][] jnodes;
    private double[][] itargetOutput;
    private double[][] irealOutput;
    //private double[][] psi;
    private double initError= .2;
    private double[][] iWeights;
    private double[][] hWeights;
    private static int cases;
    public static final int Input = 16428;
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

    public static void main(String[] args) throws IOException{
        Scanner in = new Scanner(System.in);
        File file = new File("/users/Sriram/Desktop/2014-15/Neural Nets/weightsColor.txt");
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

            //taking in the input cases
            cases=5;
            double[][] ktrainingInputs1 = new double[cases][Input];
            //initializes a double array for inputs and then puts a combination of rgb value of picture into inputs
            BufferedImage[] images = new BufferedImage[cases];
            //array of all images, decided by the input cases earlier
            for(int n =0;n<cases;n++)
            {
                images[n] = null;
                File f = new File("/users/Sriram/Desktop/2014-15/Neural Nets/image" + n + ".jpg");
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

                        //int a = (p>>24)&0xff;
                        int r = (p>>16)&0xff;
                        int g = (p>>8)&0xff;
                        int b = p & 0xff;
                        //int sum = (int)Math.round((r*0.3 + g*0.589 + b*0.11));
                        //CONSIDER PUTTING IN THE 255 inverter
                        //p = (a<<24) | (r<<16) | (g<<8) | b;
                        //p = (a<<24) | (0<<16) | (g<<8) | 0;
                        ktrainingInputs1[n][j+i*(width*3)]=r;
                        j++;
                        ktrainingInputs1[n][j+i*(width*3)]=g;
                        j++;
                        ktrainingInputs1[n][j+i*(width*3)]=b;

                        //ktrainingInputs1[n][j+i*width]=sum;

                        //Change based on what the desired input is
                        //System.out.println(g);
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
            //training the weights to then print out the weights to file
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
            System.out.println("Done");
            //autobot.printWeightMap(cases,74,74);
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

            //getting in all the inputs that will be propagated to check output
            cases=5;
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
                File f = new File("/users/Sriram/Desktop/2014-15/Neural Nets/image" + n + ".jpg");
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

                        //int a = (p>>24)&0xff;
                        int r = (p>>16)&0xff;
                        int g = (p>>8)&0xff;
                        int b = p & 0xff;
                        //int sum = (int)Math.round((r*0.3 + g*0.589 + b*0.11));

                        //p = (a<<24) | (r<<16) | (g<<8) | b;
                        //p = (a<<24) | (0<<16) | (g<<8) | 0;
                        //ktrainingInputs1[n][j+i*width]=g;
                        //ktrainingInputs1[n][j+i*width]=sum;
                        ktrainingInputs1[n][j+i*(width*3)]=r;
                        j++;
                        ktrainingInputs1[n][j+i*(width*3)]=g;
                        j++;
                        ktrainingInputs1[n][j+i*(width*3)]=b;
                        //System.out.println(g);
                        //img.setRGB(j,i,p);
                    }

                }
            }

            autobot.setInput(ktrainingInputs1);
            //setting ideal output

            autobot.propagateNet(cases); //with 5 cases
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
            //BufferedWriter bout = new BufferedWriter(new FileWriter(file2));

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

            //
            cases=5;
            CreativeArtColor autobot = new CreativeArtColor(iWeights1,hWeights1);

            //taking in the inputs
            //cases=4;
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

            int height = 74;
            int width = 74;
            //int bWidth = 100*3;
            BufferedImage[] images = new BufferedImage[cases];
            //Set to be as what the image is, can later change to be fed in.

            for(int n=0;n<cases;n++)
            {
                //File f = new File("/users/Sriram/Desktop/2014-15/Neural Nets/image.jpg");
                File f = new File("/users/Sriram/Desktop/2014-15/Neural Nets/image" + n + ".jpg");

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
                File f1 = new File("/users/Sriram/Desktop/2014-15/Neural Nets/image" + n + "Output.jpg");
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
            File f = new File("/users/Sriram/Desktop/2014-15/Neural Nets/image" + n + ".jpg");
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
            File f1 = new File("/users/Sriram/Desktop/2014-15/Neural Nets/image" + n + "WeightOutput.jpg");
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

    public void setOut()
    {
        itargetOutput = new double[][] {{1.0},{1.0},{1.0},{0.0},{1.0}};
        //itargetOutput = new double[][] {{1.0},{1.0},{1.0},{0.0},{1.0},{1.0},{1.0},{1.0},{1.0},{1.0},{1.0},{1.0}};
        //Make sure this has new outputs updated
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

    public void printStuff(int count)
    {
        int n=0;
        //File f = new File("/users/Sriram/Desktop/2014-15/Neural Nets/image.jpg");
        File f = new File("/users/Sriram/Desktop/2014-15/Neural Nets/image0.jpg");

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
        File f1 = new File("/users/Sriram/Desktop/2014-15/Neural Nets/VideoReal/" + count + "Output.jpg");
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
        yea, set an error 0 and then sum equals consistent squares of diff between target and calculated,
        uses a temp variable to hold the evaluated inputs. Correct indices.
         */
        double error = 0.0;
        //double[][] calcOutput = new double[Input][];
        for(int n=0;n<cases;n++)
        {
            //double[][] temp=jnodes;
            //calcOutput[k]=temp[Hidden-1];
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
            //THIS MAY HAVE TO BE SWITCHED BECAUSE TRAIN NEEDS TO USE PROPAGTE NET FIRST
            currError=error();
            if(currCycles%20==0) System.out.println(currError);
            currCycles++;
        }
        System.out.println("Final Error: " + currError);
        return;
    }

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
}