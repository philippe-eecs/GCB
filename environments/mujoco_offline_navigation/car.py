from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable

import os

class CarObservables(composer.Observables):

    @composer.observable
    def realsense_camera(self):
        #return observable.MJCFCamera(self._entity._mjcf_root.worldbody.body['buddy'].camera['buddy_realsense_d435i'])
        return observable.MJCFCamera(self._entity._mjcf_root.find('camera', 'buddy_realsense_d435i'), height=64, width=128)
    
    @composer.observable
    def body_position(self):
        return observable.MJCFFeature('xpos', self._entity._mjcf_root.find('body', 'buddy'))

    @composer.observable
    def body_rotation(self):
        return observable.MJCFFeature('xquat', self._entity._mjcf_root.find('body', 'buddy'))

    @composer.observable
    def sensors_framequat(self):
        return observable.MJCFFeature('sensordata',
                                    self._entity._mjcf_root.find('sensor', 'framequat'))

    def _collect_from_attachments(self, attribute_name):
        out = []
        for entity in self._entity.iter_entities(exclude_self=True):
            out.extend(getattr(entity.observables, attribute_name, []))
        return out

    @property
    def kinematic_sensors(self):
        return ([self.sensors_framequat] +
                self._collect_from_attachments('kinematic_sensors'))

class Car(composer.Robot):

    def _build(self, name='walker'):
        self._mjcf_root = mjcf.from_path(
            os.path.join('mujoco_offline_navigation', 'models', 'cars', 'pusher_car', 'buddy.xml'))
        if name:
            self._mjcf_root.model = name

        self._actuators = self.mjcf_model.find_all('actuator')

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def actuators(self):
        return self._actuators

    def apply_action(self, physics, action, random_state):
        """Apply action to walker's actuators."""
        del random_state
        physics.bind(self.actuators).ctrl = action

    def _build_observables(self):
        return CarObservables(self)
