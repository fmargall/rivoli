#ifndef BRDFSAMPLESCOUPOLE_HPP
#define BRDFSAMPLESCOUPOLE_HPP

#include <string>
#include <fstream>
#include <array>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include "logger.hpp"

typedef glm::vec2  Vector2;
typedef glm::ivec3 iVector3;

enum VERSION_BRDF_SAMPLES{BRDF_SAMPLES_V1=1, BRDF_SAMPLES_V2=2};
inline std::string versionBRDFsamplesTOstring(VERSION_BRDF_SAMPLES versionBrdfSamples){
    switch(versionBrdfSamples){
        case BRDF_SAMPLES_V1: return "v1 (θ_i/θ_o/Δφ (float) / f (float) / w (float))";
        case BRDF_SAMPLES_V2: return "v2 (to do ...)";
    }
    return "UNKNOWN";
}

inline void displayHelpBrdfSamples(){
    LOG_DEBUG("Help BRDF samples version : ", 
              versionBRDFsamplesTOstring(BRDF_SAMPLES_V1),
              versionBRDFsamplesTOstring(BRDF_SAMPLES_V2));
}

class BrdfSamplesCoupole
{
public:
    BrdfSamplesCoupole();
    ~BrdfSamplesCoupole();

    /*struct BrdfSample{
        virtual void assign(float theta_i, float theta_o, float phi_delta, float f, float w) = 0;
        virtual void get(float &theta_i, float &theta_o, float &phi_delta, float &f, float &w) = 0;
        virtual void write(std::fstream &f) = 0;
        virtual ~BrdfSample(){};
    };*/

    /*struct BrdfSamples{
        struct Sample{ };
        std::vector<std::array<std::vector<Sample*>,3>> samples;

        int getSize() { return samples.size(); }
        int getSamplesNumber(int cluster, int channel){ return samples[cluster][channel].size(); }

        void addSample(const int cluster, const int channel, const float theta_i, const float theta_o, const float phi_delta, const float f, const float w){
            Sample* s = encode(theta_i, theta_o, phi_delta, f, w);
            samples[cluster][channel].emplace_back(s);
        }
        void getSample(const int cluster, const int channel, const int sample, float &theta_i, float &theta_o, float &phi_delta, float &f, float &w){
            decode(samples[cluster][channel][sample], theta_i, theta_o, phi_delta, f, w);
        }

        virtual ~BrdfSamples(){
            for(int c=0; c<samples.size(); c++){
                for(int rgb=0; rgb<samples[c].size(); rgb++){
                    for(int s=0; s<samples[c][rgb].size(); s++){
                        delete samples[c][rgb][s];
                    }
                }
            }
        }


        virtual Sample* encode(const float theta_i, const float theta_o, const float phi_delta, const float f, const float w) = 0;
        virtual void decode(const Sample *s, float &theta_i, float &theta_o, float &phi_delta, float &f, float &w) = 0;
        virtual void writeSample(std::fstream &f, Sample *sample) = 0;
        virtual Sample* readSample(std::fstream &f) = 0;
    };*/
    struct BrdfSamples{

        virtual void addSample(const int cluster, const int channel, const float theta_i, const float theta_o, const float phi_delta, const float f, const float w) = 0;
        virtual void getSample(const int cluster, const int channel, const int sample, float &theta_i, float &theta_o, float &phi_delta, float &f, float &w) = 0;
        virtual void resize(int size) = 0;
        virtual void clearClusterSamples(int cluster) = 0;
        virtual int getSize() = 0;
        virtual int getSamplesNumber(int cluster, int channel) = 0;
        virtual int getSizeSampleStruct() = 0;
        virtual ~BrdfSamples(){};

        virtual void writeSample(std::fstream &f, const int cluster, const int rgb, const int sample) = 0;
        virtual void readSample(std::fstream &f, const int cluster, const int rgb) = 0;

    };

    struct BrdfSamplesV1 : BrdfSamples {

        struct Sample{
            float theta_i, theta_o, phi_delta, f, w;
        };

        std::vector<std::array<std::vector<Sample>,3>> samples;

        int getSize() { return samples.size(); }
        int getSamplesNumber(int cluster, int channel){ return samples[cluster][channel].size(); }
        void resize(int size){ samples.resize(size); };
        void clearClusterSamples(int cluster){ samples[cluster][0].resize(0); samples[cluster][1].resize(0); samples[cluster][2].resize(0); };
        int getSizeSampleStruct(){ return sizeof(Sample); };

        void addSample(const int cluster, const int channel, const float theta_i, const float theta_o, const float phi_delta, const float f, const float w){
            Sample s;
            s.theta_i = theta_i;
            s.phi_delta = phi_delta;
            s.theta_o = theta_o;
            s.f = f;
            s.w = w;
            samples[cluster][channel].emplace_back(s);
        }
        void getSample(const int cluster, const int channel, const int sample, float &theta_i, float &theta_o, float &phi_delta, float &f, float &w){
            Sample &s = samples[cluster][channel][sample];
            theta_i = s.theta_i;
            theta_o = s.theta_o;
            phi_delta = s.phi_delta;
            f = s.f;
            w = s.w;
        }

        void writeSample(std::fstream &f, const int cluster, const int rgb, const int sample){
            Sample &s = samples[cluster][rgb][sample];
            f.write(reinterpret_cast<char*>(&s.theta_i), sizeof(float));
            f.write(reinterpret_cast<char*>(&s.theta_o), sizeof(float));
            f.write(reinterpret_cast<char*>(&s.phi_delta), sizeof(float));
            f.write(reinterpret_cast<char*>(&s.f), sizeof(float));
            f.write(reinterpret_cast<char*>(&s.w), sizeof(float));
        }
        void readSample(std::fstream &f, const int cluster, const int rgb){
            Sample sample;
            f.read(reinterpret_cast<char*>(&sample.theta_i), sizeof(float));
            f.read(reinterpret_cast<char*>(&sample.theta_o), sizeof(float));
            f.read(reinterpret_cast<char*>(&sample.phi_delta), sizeof(float));
            f.read(reinterpret_cast<char*>(&sample.f), sizeof(float));
            f.read(reinterpret_cast<char*>(&sample.w), sizeof(float));
            samples[cluster][rgb].emplace_back(sample);
        }

        /*void encode(const float theta_i, const float theta_o, const float phi_delta, const float f, const float w, Sample &s) {
            s.theta_i = theta_i;
            s.phi_delta = phi_delta;
            s.theta_o = theta_o;
            s.f = f;
            s.w = w;
        }

        void decode(const Sample &s, float &theta_i, float &theta_o, float &phi_delta, float &f, float &w) {
            theta_i = s.theta_i;
            theta_o = s.theta_o;
            phi_delta = s.phi_delta;
            f = s.f;
            w = s.w;
        }*/
        //~BrdfSampleV1(){}
    };

    struct BrdfSamplesV2 {

        struct Sample{
        };
        //~BrdfSampleV1(){}
    };


   /* struct BrdfSampleV1 : BrdfSample {
        float _theta_i, _theta_o, _phi_delta, _f, _w;

        void encode(const float theta_i, const float theta_o, const float phi_delta, const float f, const float w) {
            _theta_i = theta_i;
            _phi_delta = phi_delta;
            _theta_o = theta_o;
            _f = f;
            _w = w;
        }
        void decode(float &theta_i, float &theta_o, float &phi_delta, float &f, float &w) {
            theta_i = _theta_i;
            theta_o = _theta_o;
            phi_delta = _phi_delta;
            f = _f;
            w = _w;
        }
        void write(std::fstream &f) {

        }
        //~BrdfSampleV1(){}
    };

    struct BrdfSampleV2 : BrdfSample {
    };*/


    /*static BrdfSample* makeBrdfSampleV1(){ return new BrdfSampleV1; }
    static BrdfSample* makeBrdfSampleV2(){ return new BrdfSampleV2; }
    BrdfSample* (*makeFunc)();*/
    //std::vector<std::array<std::vector<BrdfSample*>,3>> brdfSamples;

    std::string m_filename;
    VERSION_BRDF_SAMPLES m_version_brdf_samples;
    int m_ncluster;

    void init(VERSION_BRDF_SAMPLES version_brdf_samples, int ncluster);
    void initFromFile(std::string filename);

    std::vector<std::pair<iVector3, unsigned long long>> m_offsetReadCluster;

    /** @brief Prepare a .brdfSamples file to be read in a sparsity manner
     * @param filename: filename to read (.brdfSamples)
     * @param maxReadClusters: Max clusters to read
     * @param verbose: verbose mode
     * @param numberOfThread: Number of thread that will read the file */
    int prepareSparseRead(std::string filename, int maxReadClusters=-1, bool verbose=true, int numberOfThread=1);

    /** @brief Read one cluster after prepareSparseRead is used
     * @param cluster: ID of the cluster to read
     * @param thread: ID of the thread that read */
    void readOneCluster(int cluster, int thread=0);

    BrdfSamples *brdfSamples = nullptr;
    void write(std::string filename);

    /** @brief Get the data that has been red
     *  @param cluster : id of the cluster to read (or ID of thread when using sparse read)
    *   @param rgb : channel to get (0: red, 1: green, 2:blue)
     *  @param wo : Vector of the omega_o directions
     *  @param wi : Vector of the omega_i directions
     *  @param brdf : Vector of the reflectance value
     *  @param theta_i_max : Remove samples where theta_o > theta_i_max */
    void getData(int cluster, int rgb, std::vector<Vector2> &wo, std::vector<Vector2> &wi, std::vector<double> &brdf, float theta_i_max = glm::half_pi<float>());

    void display();

    /*void addSamples(int cluster, int channel, float theta_i, float theta_o, float phi_delta, float f, float w);

    void init(VERSION_BRDF_SAMPLES version_brdf_samples, int ncluster);
    void initFromFile(std::string filename);
    void write(std::string filename);*/

};

#endif // BRDFSAMPLESCOUPOLE_HPP
