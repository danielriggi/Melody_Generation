![](images/denon-dj-prime-4-controller-main1.jpg)
 # Generate EDM Melodies Using Recurrent Neural Networks

 ## Background  

 Despite lots of time and effort, my attempts at making dance music have been decidedly unsuccesful. Perhaps a computer can do a better job for me! 

 ## Data
Music is encoded in a number of different ways. For this project I chose to use Musical Instrument Digital Interface (MIDI) files. From Wikipedia, the definition of MIDI is: 

 "MIDI is a technical standard that describes a communications protocol, digital interface, and electrical connectors that connect a wide variety of electronic musical instruments, computers, and related audio devices for playing, editing and recording music.[1] A single MIDI link through a MIDI cable can carry up to sixteen channels of information, each of which can be routed to a separate device or instrument. This could be sixteen different digital instruments, for example. 

 I gathered EDM MIDI's from <a href="https://www.nonstop2k.com/"> nonstop2k </a>, <a href=https://www.cprato.com/> Carlos's MIDI </a>, and <a href=https://bitmidi.com/> bitmidi</a>. In total I collected 71 MIDIs from artists including Marshmellow, Alesso, Avicii, Tiesto, Kygo, and more.

 ## Pre-processing

 I chose to use the pretty_midi library within python to pre-process MIDI's. Below is an image of a pretty_midi object and how it interprets a MIDI file:

 ![](images/midiscreenshot.png)

 As you can see, a MIDI basically has four components: Note on, Note off, pitch, and velocity. Pitch is the number of the note played and velocity is the force with which the note is played (think pressing a piano key). Below is a visual representation of how MIDI corresponds to notes:

 ![](images/miditopiano.png)

 My original intent was to generate entire songs. I soon realized I was doomed when nearly every I researched on medium and towardsdatascience focused on classical piano music. This is for good reason, generating music with multiple instruments is difficult. Basically, it would involve training a model to produce melodies. and then training another model to accompany that model. One possible approach to this is generating audio with other raw audio, such as <a href=https://deepmind.com/blog/article/wavenet-generative-model-raw-audio> WaveNet </a>. This means I had to extract the melodies from my MIDI's. This, predictably, resulted in significant loss. Melodies in multi-instrument songs interact with the accompaniments. When you remove them, it fundamentally changes the melody. To attempt to mitigate this, I removed long rests that existed in the melodies alone. Below is an example of how the sheet music looked before and after processing:

 ### Before:

 ![](images/tiestobefore.com.png)

 ### After:

 ![](images/tiestoafter.com.png)


