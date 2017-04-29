/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 300;
	
	default_random_engine gen;
	normal_distribution<double> dist_x(x,std[0]);
	normal_distribution<double> dist_y(y,std[1]);
	normal_distribution<double> dist_theta(theta,std[2]);

	for(int i = 0; i < num_particles; i++){
		Particle p;
		p.id     = i;
		p.x      = dist_x(gen);
		p.y      = dist_y(gen);
		p.theta  = dist_theta(gen);
		p.weight = 1.0f;
		particles.push_back(p);
	}

	weights.resize(num_particles, 1.0f);
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;
	normal_distribution<double> dist_x(0,std_pos[0]);
	normal_distribution<double> dist_y(0,std_pos[1]);
	normal_distribution<double> dist_yaw(0,std_pos[2]);

	double x_f, y_f, theta_f;
	
	for(int i = 0; i < num_particles; i++){
		
		Particle c_particle = particles[i];
		double x_o = c_particle.x;
		double y_o = c_particle.y;
		double theta_o = c_particle.theta;

		if (fabs(yaw_rate) < 0.01){
			x_f = x_o + velocity * delta_t * cos(theta_o) + dist_x(gen);
			y_f = y_o + velocity * delta_t * sin(theta_o) + dist_y(gen);
			theta_f = theta_o + dist_yaw(gen);
		}else{
			x_f = x_o + velocity / yaw_rate * (sin(theta_o + yaw_rate * delta_t) - sin(theta_o)) + dist_x(gen);
			y_f = y_o + velocity / yaw_rate * (cos(theta_o) - cos(theta_o + yaw_rate * delta_t)) + dist_y(gen);
			theta_f = theta_o + yaw_rate * delta_t + dist_yaw(gen);
		}
		//cout << "X:"<< xf << " Y:" << yf << endl;
		particles[i].x = x_f;
		particles[i].y = y_f;
		particles[i].theta = theta_f; 
	}
}

std::vector<LandmarkObs> ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	double min_dist, dist, dx, dy;
	int min_id;

	std::vector<LandmarkObs> predicted_observation;

	for(int i =0; i < observations.size(); i++){
		LandmarkObs c_obs = observations[i];
		LandmarkObs p_obs;
		min_dist = INFINITY;
		min_id = -1;
		for(int j = 0; j < predicted.size(); j++){
			LandmarkObs c_pred = predicted[j];
			
			dx = (c_obs.x - c_pred.x);
			dy = (c_obs.y - c_pred.y);
			dist = sqrt(dx*dx + dy*dy);

			if(dist < min_dist){
				min_dist = dist;
				p_obs = c_pred;
			}
		}
		predicted_observation.push_back(p_obs);
	}

	return predicted_observation;		
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	default_random_engine gen;
	normal_distribution<double> dx(0,std_landmark[0]);
	normal_distribution<double> dy(0,std_landmark[1]);

	// bring observations into global coordinate in the particles' perspective
	for(int i = 0; i < num_particles; i++){

		Particle c_particle = particles[i];
		double tx = c_particle.x;
		double ty = c_particle.y;
		double theta = c_particle.theta;

		std::vector<LandmarkObs> predicted_landmarks;

		for(int j = 0; j < map_landmarks.landmark_list.size(); j++){
			Map::single_landmark_s c_landmark = map_landmarks.landmark_list[j];
			LandmarkObs temp_landmarks;

			double land_x = c_landmark.x_f;
			double land_y = c_landmark.y_f;
			int land_id = c_landmark.id_i;
			
			double dx = c_particle.x - land_x;
			double dy = c_particle.y - land_y;
			double dist = sqrt(dx*dx + dy*dy);
			
			if(dist <= sensor_range){
				temp_landmarks = {land_id, land_x, land_y};
				predicted_landmarks.push_back(temp_landmarks);
			}
		}

		std::vector<LandmarkObs> transformed_observations;
		
		for(int k = 0; k < observations.size(); k++){
			LandmarkObs temp_obs;
			double trans_x = observations[k].x * cos(theta) - observations[k].y * sin(theta) + tx;
			double trans_y = observations[k].x * sin(theta) + observations[k].y * cos(theta) + ty;
			
			temp_obs = {observations[k].id, trans_x, trans_y};
			transformed_observations.push_back(temp_obs);
		}
		
		std::vector<LandmarkObs> predict_obs = dataAssociation(predicted_landmarks, transformed_observations);

		double weight = 1.0;
		double den = (2.0 * M_PI * std_landmark[0] * std_landmark[1]);
		for(int l = 0; l < transformed_observations.size(); l++){
			LandmarkObs c_obs = transformed_observations[l]; //current transformed observation
			LandmarkObs a_obs = predict_obs[l];				 //current associated observation
			double x_num = ((c_obs.x - a_obs.x) * (c_obs.x - a_obs.x)) / (2 * std_landmark[0] * std_landmark[0]);
			double y_num = ((c_obs.y - a_obs.y) * (c_obs.y - a_obs.y)) / (2 * std_landmark[1] * std_landmark[1]);				
			double P = exp(-(x_num+y_num))/den;
			//cout << "x_num:" << x_num << endl;
			//cout << "x:" << transformed_observations[l].x << "Px:" << predicted_landmarks[m].x << endl;
			
			//cout << "y_num:" << y_num << endl;
			//cout << "y:" << transformed_observations[l].y << "Py:" << predicted_landmarks[m].y << endl;
			
			//cout << "P" << P << endl;
			weight = weight * P;
		}
		//cout << weight << endl;
		weights[i] = weight;
		particles[i].weight = weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> new_particles;
	default_random_engine gen;
	discrete_distribution<int> d(weights.begin(), weights.end());
	for(int i = 0;  i < num_particles; i++){
        new_particles.push_back(particles[d(gen)]);
    }
    particles = new_particles;

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
