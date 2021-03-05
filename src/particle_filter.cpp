/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::cout;
using std::endl;

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  
  std::default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
    
  for(int i=0; i<num_particles; ++i){
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1;
    
    particles.push_back(particle);
    weights.push_back(particle.weight);
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  double x0, y0, theta0, xf, yf, thetaf;
  
  std::default_random_engine gen;   
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i=0; i<num_particles; ++i){
    x0 = particles[i].x;
    y0 = particles[i].y;
    theta0 = particles[i].theta;

    if (fabs(yaw_rate) < 0.00001){
      xf = x0 + velocity * delta_t * cos(theta0);
      yf = y0 + velocity * delta_t * sin(theta0);
      thetaf = theta0;
    }
    else{
      xf = x0 + velocity/yaw_rate * (sin(theta0 + yaw_rate * delta_t) - sin(theta0));
      yf = y0 + velocity/yaw_rate * (cos(theta0) - cos(theta0 + yaw_rate * delta_t));
      thetaf = theta0 + yaw_rate * delta_t;
    }
    
    particles[i].x = xf + dist_x(gen);
    particles[i].y = yf + dist_y(gen);
    particles[i].theta = thetaf + dist_theta(gen);
  }
}


void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (unsigned int i=0; i<observations.size(); ++i){
    double obsx = observations[i].x;
    double obsy = observations[i].y;
    double min = INFINITY;
    observations[i].id = -1;
    
    for(unsigned int j=0; j<predicted.size(); ++j){
      double predx = predicted[j].x;
      double predy = predicted[j].y;
      double distance = dist(obsx, obsy, predx, predy);
      
      if (distance < min){
        observations[i].id = j;
        min = distance;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */ 
  
  std::vector<Map::single_landmark_s> landmarks = map_landmarks.landmark_list;

  for (int i=0; i<num_particles; ++i){
    double x_part = particles[i].x;
    double y_part = particles[i].y;
    double theta = particles[i].theta;
    
    vector<LandmarkObs> predicted;   
    for(unsigned int k=0; k < landmarks.size(); ++k){
      double distance = dist(x_part, y_part, landmarks[k].x_f, landmarks[k].y_f); 
      if (distance <= sensor_range){
        predicted.push_back(LandmarkObs{landmarks[k].id_i, landmarks[k].x_f, landmarks[k].y_f});
      }  
    }

    vector<LandmarkObs> obs_map;
    for (unsigned int j=0; j<observations.size(); ++j){
      double x_obs = observations[j].x;
      double y_obs = observations[j].y;

      double x_map = x_part + (cos(theta) * x_obs) - (sin(theta) * y_obs);
      double y_map = y_part + (sin(theta) * x_obs) + (cos(theta) * y_obs);

      obs_map.push_back({observations[j].id, x_map, y_map});
    }
    
    dataAssociation(predicted, obs_map);

    double weight = 1;
    for (unsigned int n=0; n<obs_map.size(); ++n){
      weight = weight * multiv_prob(std_landmark[0], std_landmark[1], 
                                    obs_map[n].x, obs_map[n].y, predicted[obs_map[n].id].x, predicted[obs_map[n].id].y);
    }

    particles[i].weight = weight;
    weights[i] = weight;
  }
}


void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> new_particles(num_particles);
  
  std::default_random_engine gen;
  std::discrete_distribution<int> dist(weights.begin(), weights.end()); // discrete distribution based on weights
  
  for(int i=0; i<num_particles; ++i){
    new_particles[i] = particles[dist(gen)];
  }
  
  particles = new_particles;
  
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

double ParticleFilter::multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y){
  // calculate normalization term
  double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
               + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
    
  // calculate weight using normalization terms and exponent
  double weight;
  weight = gauss_norm * exp(-exponent);
    
  return weight;
}