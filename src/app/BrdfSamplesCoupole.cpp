#include <iostream>

#include "BrdfSamplesCoupole.hpp"

BrdfSamplesCoupole::BrdfSamplesCoupole()
{

}

BrdfSamplesCoupole::~BrdfSamplesCoupole()
{
    if(brdfSamples!=nullptr) delete brdfSamples;
}

void BrdfSamplesCoupole::init(VERSION_BRDF_SAMPLES version_brdf_samples, int ncluster){

    m_ncluster = ncluster;
    m_version_brdf_samples=version_brdf_samples;


    switch(version_brdf_samples){
    case BRDF_SAMPLES_V1: brdfSamples = new BrdfSamplesV1; break;
    case BRDF_SAMPLES_V2:  break;
    }

    brdfSamples->resize(ncluster);

}

void BrdfSamplesCoupole::initFromFile(std::string filename){
    if(brdfSamples!=nullptr) delete brdfSamples;

    m_filename = filename;
    std::fstream f;
    f.open(filename, std::fstream::in | std::fstream::binary);
    f.read(reinterpret_cast<char*>(&m_version_brdf_samples), sizeof(int));
    f.read(reinterpret_cast<char*>(&m_ncluster), sizeof(int));

    init(m_version_brdf_samples, m_ncluster);

    LOG_INFO("BrdfSamplesCoupole read : ", filename, m_version_brdf_samples, " ", m_ncluster);

    std::vector<iVector3> samples(m_ncluster);
    f.read(reinterpret_cast<char*>(samples.data()), sizeof(iVector3) * m_ncluster);

	LOG_DEBUG("Reading clusters...");
    for(int c=0; c<brdfSamples->getSize(); c++){
        for(int rgb=0; rgb<3; rgb++){
            //int nsamples;
            //f.read(reinterpret_cast<char*>(&nsamples), sizeof(int));
            for(int s=0; s<samples[c][rgb]; s++){
                brdfSamples->readSample(f, c, rgb);
            }
        }
		logger.displayProgressBar(c, brdfSamples->getSize());
    }
	LOG_DEBUG("Reading clusters done. ", brdfSamples->getSize(), " found.");
    f.close();
}


int BrdfSamplesCoupole::prepareSparseRead(std::string filename, int maxReadClusters, bool verbose, int numberOfThread){
    if(brdfSamples!=nullptr) delete brdfSamples;

    m_filename = filename;
    std::fstream f;
    f.open(filename, std::fstream::in | std::fstream::binary);
    f.read(reinterpret_cast<char*>(&m_version_brdf_samples), sizeof(int));
    int ncluster;
    f.read(reinterpret_cast<char*>(&ncluster), sizeof(int));

    init(m_version_brdf_samples, numberOfThread);

    LOG_DEBUG("BrdfSamplesCoupole sparse read preparation : ", filename);
    LOG_DEBUG(m_version_brdf_samples, " ", m_ncluster, " ", ncluster);

    std::vector<iVector3> samples(ncluster);
    f.read(reinterpret_cast<char*>(samples.data()), sizeof(iVector3) * ncluster);

    int SampleStructSize = brdfSamples->getSizeSampleStruct();
    unsigned long long totalDecal = 2 * sizeof(int) + ncluster * sizeof(iVector3);

    if(maxReadClusters!=-1) ncluster=std::min(maxReadClusters, ncluster);

    m_offsetReadCluster.resize(ncluster);
    LOG_DEBUG("Reading clusters...");
    for(int c=0; c<ncluster; c++){
        m_offsetReadCluster[c] = {samples[c], totalDecal};
        for(int rgb=0; rgb<3; rgb++){
            //int nsamples;
            //f.read(reinterpret_cast<char*>(&nsamples), sizeof(int));
            totalDecal +=samples[c][rgb] * SampleStructSize;
            //f.seekg(nsamples * SampleStructSize, std::ios_base::cur);
        }
        logger.displayProgressBar(c, ncluster);
    }
	LOG_DEBUG("Reading clusters done. ", ncluster, " found.");
    f.close();
    return ncluster;
}

void BrdfSamplesCoupole::readOneCluster(int cluster, int thread){

    if(cluster<0 || cluster>=m_offsetReadCluster.size())
        LOG_CRITICAL("Cluster : ", cluster, " not in range");
    
    std::fstream f;
    f.open(m_filename, std::fstream::in | std::fstream::binary);

    f.seekg(m_offsetReadCluster[cluster].second);
    brdfSamples->clearClusterSamples(thread);
    for(int rgb=0; rgb<3; rgb++){
        int nsamples = m_offsetReadCluster[cluster].first[rgb];
        //f.read(reinterpret_cast<char*>(&nsamples), sizeof(int));
        for(int s=0; s<nsamples; s++){
            brdfSamples->readSample(f, thread, rgb);
        }
    }

    f.close();
}

void BrdfSamplesCoupole::write(std::string filename){
    std::fstream f;
    LOG_DEBUG("Writing: ", filename);
    f.open(filename, std::fstream::out | std::fstream::binary);
    f.write(reinterpret_cast<char*>(&m_version_brdf_samples), sizeof(int));
    f.write(reinterpret_cast<char*>(&m_ncluster), sizeof(int));

    for(int c=0; c<brdfSamples->getSize(); c++){
        for(int rgb=0; rgb<3; rgb++){
            int nsamples = brdfSamples->getSamplesNumber(c, rgb);
            f.write(reinterpret_cast<char*>(&nsamples), sizeof(int));
        }
    }

    for(int c=0; c<brdfSamples->getSize(); c++){
        for(int rgb=0; rgb<3; rgb++){
            int nsamples = brdfSamples->getSamplesNumber(c, rgb);
            //f.write(reinterpret_cast<char*>(&nsamples), sizeof(int));
            for(int s=0; s<nsamples; s++){
                brdfSamples->writeSample(f, c, rgb, s);
            }
        }
        std::cout << "\r              Save clusters: " << c+1  << "/" << brdfSamples->getSize() << "                  " << std::flush;
    }
    std::cout << std::endl;

    f.close();
}

void BrdfSamplesCoupole::getData(int cluster, int rgb, std::vector<Vector2> &wo, std::vector<Vector2> &wi, std::vector<double> &brdf, float theta_i_max){
    wo.resize(0);
    wi.resize(0);
    brdf.resize(0);
    int nsamples = brdfSamples->getSamplesNumber(cluster, rgb);
    for(int s=0; s<nsamples; s++){
        float theta_i, theta_o, phi_delta, f, w;
        brdfSamples->getSample(cluster, rgb, s, theta_i, theta_o, phi_delta, f, w);
        if(theta_i < theta_i_max){
            wo.push_back(Vector2(theta_o,0));
            wi.push_back(Vector2(theta_i, phi_delta));
            brdf.push_back(f);
        }
    }
}

void BrdfSamplesCoupole::display(){
    LOG_DEBUG("Brdf samples :Version : ", versionBRDFsamplesTOstring(m_version_brdf_samples));
}
