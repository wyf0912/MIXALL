from tensorboardX import SummaryWriter
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from data_process.data_gen_VD import *
from utils import *
from networks import *
import datetime
import os
import copy
import sklearn
import heapq


class ModelBaseline_VD_Mixup:
    def __init__(self, flags):
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.configure(flags)
        self.setup_path(flags)
        self.init_network_parameter(flags)

        if not os.path.exists(flags.logs):
            os.mkdir(flags.logs)
        if not os.path.exists(flags.model_path):
            os.mkdir(flags.model_path)

    def __del__(self):
        print('release source')

    def configure(self, flags):
        self.flags = flags
        self.flags_log = os.path.join(flags.logs, '%s %.2f.txt' % (
            flags.method + '_mixup' + '_multi_feature' if flags.mixup_method else '', self.flags.mixup_alpha))
        self.model_store = os.path.join(flags.model_path,
                                        '%s %.2f.pkl' % (flags.method + '_mixup_feature', self.flags.mixup_alpha))
        self.activate_load_model = True
        self.writer = SummaryWriter()

    def setup_path(self, flags):
        self.best_accuracy_val = -1
        if flags.dataset == 'VD':
            self.domains_name = get_domain_name()
            data_folder, train_data, val_data, test_data = get_data_folder()
        else:
            assert flags.dataset == 'VD', 'The current heterogeous DG code uses VD dataset'
        self.train_paths = []
        for data in train_data:
            path = os.path.join(data_folder, data)
            self.train_paths.append(path)

        self.val_paths = []
        for data in val_data:
            path = os.path.join(data_folder, data)
            self.val_paths.append(path)

        self.test_paths = []
        for data in test_data:
            path = os.path.join(data_folder, data)
            self.test_paths.append(path)

        unseen_index = 6
        self.unseen_data_path = []
        index = unseen_index
        for data in test_data[unseen_index:]:
            path = os.path.join(data_folder, data)
            self.unseen_data_path.append(self.train_paths[index])
            self.unseen_data_path.append(self.val_paths[index])
            self.train_paths.remove(self.train_paths[index])
            self.val_paths.remove(self.val_paths[index])

        if not os.path.exists(flags.logs):
            os.mkdir(flags.logs)
        flags_log = os.path.join(flags.logs, 'path_log.txt')
        write_log(str(self.train_paths), flags_log)
        write_log(str(self.val_paths), flags_log)
        write_log(str(self.unseen_data_path), flags_log)

        self.batImageGenTrains = []
        for train_path in self.train_paths:
            batImageGenTrain = BatchImageGenerator(flags=flags, file_path=train_path, stage='train',
                                                   metatest=False, b_unfold_label=False)
            self.batImageGenTrains.append(batImageGenTrain)

        self.batImageGenTrains_metatest = []
        for train_path in self.train_paths:
            batImageGenTrain_metatest = BatchImageGenerator(flags=flags, file_path=train_path, stage='train',
                                                            metatest=True, b_unfold_label=False)
            self.batImageGenTrains_metatest.append(batImageGenTrain_metatest)

        self.batImageGenVals = []
        for val_path in self.val_paths:
            batImageGenVal = BatchImageGenerator(flags=flags, file_path=val_path, stage='val',
                                                 metatest=False, b_unfold_label=True)
            self.batImageGenVals.append(batImageGenVal)

        self.batImageGenTests = []
        for test_path in self.test_paths:
            batImageGenTest = BatchImageGenerator(flags=flags, file_path=test_path, stage='test',
                                                  metatest=False, b_unfold_label=False)
            self.batImageGenTests.append(batImageGenTest)

    def init_network_parameter(self, flags):
        self.weight_decay = 1e-4  # 3e-4
        self.batch_size = flags.batch_size

        self.h = 512  # 1000
        self.hh = 100
        self.num_domain = 10
        self.num_test_domain = 4
        self.num_train_domain = self.num_domain - self.num_test_domain
        ######################################################
        self.feature_extractor_network = resnet18(pretrained=True)
        self.param_optim_theta = freeze_layer(self.feature_extractor_network)
        # theta means the network parameter of feature extractor, from d (the size of input) to h(the size of feature layer).
        self.opt = torch.optim.Adam(self.param_optim_theta, lr=flags.lr, amsgrad=True, weight_decay=self.weight_decay)
        # phi means the classifer network parameter, from h (the output feature layer of input data) to c (the number of classes).

        self.phi_all = classifier(
            100 + 2 + 43 + 1623 + 10 + 1000)  # CIFAR-100  Daimler Ped GTSRB Omniglot SVHN ImageNet
        self.label_offset = [0, 100, 102, 145, 145 + 1623, 145 + 1633]
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.opt_phi = torch.optim.Adam(self.phi_all.parameters(), lr=flags.lr, amsgrad=True,
                                        weight_decay=self.weight_decay)

    def load_state_dict(self, state_dict=''):
        tmp = torch.load(state_dict)
        pretrained_dict = tmp[0]
        # load the new state dict
        self.feature_extractor_network.load_state_dict(pretrained_dict)
        self.phi_all.load_state_dict(tmp[1])

    def mixup_data(self, x, y, alpha=1, method='multi'):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if method == 'multi':
            batch_size = x.size()[0]
            lam = torch.rand(6, batch_size).cuda()
            lam = F.softmax(lam * self.flags.mixup_alpha, dim=0)
            index_list = []
            mixed_x = 0
            for i in range(6):
                index_list.append(torch.randperm(batch_size).cuda())
                if self.flags.mix_from =='image':
                    mixed_x += x[index_list[i], :] * lam[i].reshape(batch_size, 1, 1, 1)
                else:
                    mixed_x += x[index_list[i], :] * lam[i].reshape(batch_size, 1)
            return mixed_x, index_list, y, lam

        else:
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1
            batch_size = x.size()[0]
            index = torch.randperm(batch_size).cuda()
            mixed_x = lam * x + (1 - lam) * x[index, :]
            y_a, y_b = y, y[index]
            return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, pred, lam, y_a, y_b, method=None):
        if method == 'multi':
            index_list, y, loss = y_a, y_b, 0
            for i in range(6):
                loss += lam[i] * F.cross_entropy(pred, y[index_list[i]], reduction="none")
            return loss.mean()
        else:
            return lam * self.ce_loss(pred, y_a) + (1 - lam) * self.ce_loss(pred, y_b)

    def heldout_test(self, flags, filename=None, iter=None):
        # load the best model on the validation data
        assert not (filename and iter)
        filename = 'best_model_mixup 0.600000 iter:6000.tar' if not iter else None
        if filename and iter is None:
            print('loading the model from file for held out test ')
            model_path = os.path.join(flags.model_path, filename)
            self.load_state_dict(state_dict=model_path)

        # Set the svm parameters by cross-validation
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        for i in range(self.num_test_domain):
            self.feature_extractor_network.eval()
            # test domains
            clf = GridSearchCV(svm.SVC(), tuned_parameters, scoring='precision_macro', n_jobs=5)

            batImageGenTest_train = BatchImageGenerator(flags=flags, file_path=self.unseen_data_path[2 * i],
                                                        stage='test', metatest=False, b_unfold_label=False)
            images_train = batImageGenTest_train.images
            labels_train = batImageGenTest_train.labels
            threshold = 100
            if len(images_train) > threshold:

                n_slices_test = len(images_train) // threshold
                indices_test = []
                for per_slice in range(n_slices_test - 1):
                    indices_test.append(len(images_train) * (per_slice + 1) // n_slices_test)
                train_image_splits = np.split(images_train, indices_or_sections=indices_test)

            # Verify the splits are correct
            train_image_splits_2_whole = np.concatenate(train_image_splits)
            assert np.all(images_train == train_image_splits_2_whole)

            # split the test data into splits and test them one by one
            train_feature_output = []
            for train_image_split in train_image_splits:
                # print(len(test_image_split))
                train_image_split = get_image(train_image_split)
                # print (test_image_split[0].shape)
                train_image_split = torch.from_numpy(np.array(train_image_split, dtype=np.float32))
                train_image_split = Variable(train_image_split, requires_grad=False).cuda()

                feature_out = self.feature_extractor_network(train_image_split).data.cpu().numpy()
                train_feature_output.append(feature_out)

            # concatenate the test predictions first
            train_feature_output = np.concatenate(train_feature_output)
            clf.fit(train_feature_output, labels_train)
            torch.cuda.empty_cache()
            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            write_log('Best parameters set found on development set:', self.flags_log)
            write_log(clf.best_params_, self.flags_log)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()
            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")

            batImageGenTest_test = BatchImageGenerator(flags=flags, file_path=self.unseen_data_path[2 * i + 1],
                                                       stage='test', metatest=False, b_unfold_label=False)

            images_test = batImageGenTest_test.images
            labels_test = batImageGenTest_test.labels
            threshold = 100
            if len(images_test) > threshold:

                n_slices_test = len(images_test) // threshold
                indices_test = []
                for per_slice in range(n_slices_test - 1):
                    indices_test.append(len(images_test) * (per_slice + 1) // n_slices_test)
                test_image_splits = np.split(images_test, indices_or_sections=indices_test)

            # split the test data into splits and test them one by one
            test_classifier_output = []
            for test_image_split in test_image_splits:
                # print(len(test_image_split))
                test_image_split = get_image(test_image_split)
                # print (test_image_split[0].shape)
                test_image_split = torch.from_numpy(np.array(test_image_split, dtype=np.float32))
                test_image_split = Variable(test_image_split, requires_grad=False).cuda()
                feature_out = self.feature_extractor_network(test_image_split)

                classifier_out = clf.predict(feature_out.data.cpu().numpy())
                test_classifier_output.append(classifier_out)
            test_classifier_output = np.concatenate(test_classifier_output)
            torch.cuda.empty_cache()
            accuracy = classification_report(labels_test, test_classifier_output)
            print(accuracy)
            precision = np.mean(test_classifier_output == labels_test)
            print(precision)
            # accuracy
            accuracy_info = 'the test domain %s.\n' % (self.domains_name[str(i + self.num_train_domain)])
            flags_log = os.path.join(flags.logs, 'heldout_test_log_%s' % filename) if filename else os.path.join(
                flags.logs,
                'heldout_test_log_mixup %s:%.2f iter:%d.txt' % (self.flags.mixup_method, flags.mixup_alpha, iter))
            write_log(accuracy_info, flags_log)
            write_log(clf.best_params_, flags_log)
            # write_log(accuracy, flags_log)
            write_log(precision, flags_log)
        self.writer.close()

    def heldout_test_knn(self, flags, filename=None, iter=None):
        # load the best model on the validation data
        assert not (filename and iter)
        filename = 'best_model_mixup 0.40 iter:19000.tar' if not iter else None
        # filename = 'best_model_transfered.tar' if not iter else None
        # best_model_mixup 0.400000 iter:9500.tar
        if filename and iter is None:
            model_path = os.path.join(flags.model_path, filename)
            self.load_state_dict(state_dict=model_path)

        # Set the svm parameters by cross-validation
        tuned_parameters = [
            {
                # 'weights': [cos_dist],#[sklearn.metrics.pairwise.cosine_similarity],
                'n_neighbors': [i for i in range(1, 21)],
                'metric': ["euclidean"]  # [cos_dist] # [sklearn.metrics.pairwise.cosine_similarity] #[cos_dist]
            },
            # {
            #     'weights':['distance'],
            #     'n_neighbors':[i for i in range(1,11)],
            #     'p':[i for i in range(1,6)]
            # }
        ]
        for i in range(self.num_test_domain):
            self.feature_extractor_network.eval()
        # test domains
        for i in range(self.num_test_domain):
            self.feature_extractor_network.eval()
            # test domains
            clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, scoring='precision_macro', n_jobs=10)

            batImageGenTest_train = BatchImageGenerator(flags=flags, file_path=self.unseen_data_path[2 * i],
                                                        stage='test', metatest=False, b_unfold_label=False)
            images_train = batImageGenTest_train.images
            labels_train = batImageGenTest_train.labels
            threshold = 100  # it will left a single sample if use 100 which will stop the process
            if len(images_train) > threshold:

                n_slices_test = len(images_train) // threshold
                indices_test = []
                for per_slice in range(n_slices_test - 1):
                    indices_test.append(len(images_train) * (per_slice + 1) // n_slices_test)
                train_image_splits = np.split(images_train, indices_or_sections=indices_test)

            # Verify the splits are correct
            train_image_splits_2_whole = np.concatenate(train_image_splits)
            assert np.all(images_train == train_image_splits_2_whole)

            # split the test data into splits and test them one by one
            train_feature_output = []
            for train_image_split in train_image_splits:
                # print(len(test_image_split))
                train_image_split = get_image(train_image_split)
                # print (test_image_split[0].shape)
                train_image_split = torch.from_numpy(np.array(train_image_split, dtype=np.float32))
                train_image_split = Variable(train_image_split, requires_grad=False).cuda()

                feature_out = self.feature_extractor_network(train_image_split).data.cpu().numpy()
                train_feature_output.append(feature_out)

            # concatenate the test predictions first
            train_feature_output = np.concatenate(train_feature_output)
            train_feature_output = sklearn.preprocessing.normalize(train_feature_output, norm='l2')
            clf.fit(train_feature_output, labels_train.reshape(-1, 1))
            torch.cuda.empty_cache()
            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            write_log('Best parameters set found on development set:', self.flags_log)
            write_log(clf.best_params_, self.flags_log)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()
            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")

            batImageGenTest_test = BatchImageGenerator(flags=flags, file_path=self.unseen_data_path[2 * i + 1],
                                                       stage='test', metatest=False, b_unfold_label=False)

            images_test = batImageGenTest_test.images
            labels_test = batImageGenTest_test.labels
            threshold = 100
            if len(images_test) > threshold:

                n_slices_test = len(images_test) // threshold
                indices_test = []
                for per_slice in range(n_slices_test - 1):
                    indices_test.append(len(images_test) * (per_slice + 1) // n_slices_test)
                test_image_splits = np.split(images_test, indices_or_sections=indices_test)

            # split the test data into splits and test them one by one
            test_classifier_output = []
            for test_image_split in test_image_splits:
                # print(len(test_image_split))
                test_image_split = get_image(test_image_split)
                # print (test_image_split[0].shape)
                test_image_split = torch.from_numpy(np.array(test_image_split, dtype=np.float32))
                test_image_split = Variable(test_image_split, requires_grad=False).cuda()
                feature_out = self.feature_extractor_network(test_image_split)

                classifier_out = clf.predict(feature_out.data.cpu().numpy())
                test_classifier_output.append(classifier_out)
            test_classifier_output = np.concatenate(test_classifier_output)
            torch.cuda.empty_cache()
            accuracy = classification_report(labels_test, test_classifier_output)
            print(accuracy)
            precision = np.mean(test_classifier_output == labels_test)
            print(precision)
            # accuracy
            accuracy_info = 'the test domain %s.\n' % (self.domains_name[str(i + self.num_train_domain)])
            flags_log = os.path.join(flags.logs, 'heldout_knn_test_log.txt' if filename else os.path.join(
                flags.logs,
                'heldout_knn_test_log.txt'))
            write_log(accuracy_info, flags_log)
            write_log(clf.best_params_, flags_log)
            # write_log(accuracy, flags_log)
            write_log(precision, flags_log)
        self.writer.close()

    def heldout_test_knn_cos(self, flags, filename=None, iter=None):
        # load the best model on the validation data
        assert not (filename and iter)
        filename = 'best_model_mixup multi 6.00 iter:16000.tar' if not iter else None
        #filename = 'best_model_transfered.tar' if not iter else None
        # best_model_mixup 0.400000 iter:9500.tar
        if filename and iter is None:
            model_path = os.path.join(flags.model_path, filename)
            self.load_state_dict(state_dict=model_path)

        for i in range(self.num_test_domain):
            self.feature_extractor_network.eval()
        # test domains
        for i in range(self.num_test_domain):
            self.feature_extractor_network.eval()
            # test domains
            batImageGenTest_train = BatchImageGenerator(flags=flags, file_path=self.unseen_data_path[2 * i],
                                                        stage='test', metatest=False, b_unfold_label=False)
            images_train = batImageGenTest_train.images
            labels_train = batImageGenTest_train.labels
            threshold = 100  # it will left a single sample if use 100 which will stop the process
            if len(images_train) > threshold:

                n_slices_test = len(images_train) // threshold
                indices_test = []
                for per_slice in range(n_slices_test - 1):
                    indices_test.append(len(images_train) * (per_slice + 1) // n_slices_test)
                train_image_splits = np.split(images_train, indices_or_sections=indices_test)

            # Verify the splits are correct
            train_image_splits_2_whole = np.concatenate(train_image_splits)
            assert np.all(images_train == train_image_splits_2_whole)

            # split the test data into splits and test them one by one
            train_feature_output = []
            for train_image_split in train_image_splits:
                # print(len(test_image_split))
                train_image_split = get_image(train_image_split)
                # print (test_image_split[0].shape)
                train_image_split = torch.from_numpy(np.array(train_image_split, dtype=np.float32))
                train_image_split = Variable(train_image_split, requires_grad=False).cuda()

                feature_out = self.feature_extractor_network(train_image_split).data.cpu().numpy()
                train_feature_output.append(feature_out)

            # concatenate the test predictions first
            train_feature_output = np.concatenate(train_feature_output)
            torch.cuda.empty_cache()

            batImageGenTest_test = BatchImageGenerator(flags=flags, file_path=self.unseen_data_path[2 * i + 1],
                                                       stage='test', metatest=False, b_unfold_label=False)

            images_test = batImageGenTest_test.images
            labels_test = batImageGenTest_test.labels
            threshold = 100
            if len(images_test) > threshold:

                n_slices_test = len(images_test) // threshold
                indices_test = []
                for per_slice in range(n_slices_test - 1):
                    indices_test.append(len(images_test) * (per_slice + 1) // n_slices_test)
                test_image_splits = np.split(images_test, indices_or_sections=indices_test)

            # split the test data into splits and test them one by one
            feature_out_list = []
            for test_image_split in test_image_splits:
                # print(len(test_image_split))
                test_image_split = get_image(test_image_split)
                # print (test_image_split[0].shape)
                test_image_split = torch.from_numpy(np.array(test_image_split, dtype=np.float32))
                test_image_split = Variable(test_image_split, requires_grad=False).cuda()
                feature_out = self.feature_extractor_network(test_image_split).data.cpu().numpy()
                feature_out_list.append(feature_out)

            feature_out_list = np.concatenate(feature_out_list)
            torch.cuda.empty_cache()
            accuracy = []
            for k in range(1, 21):
                accuracy.append(self.cos_knn(k, feature_out_list, labels_test, train_feature_output, labels_train))
                #print(accuracy[k-1])
            print(self.domains_name[str(i + self.num_train_domain)], accuracy, max(accuracy))
            self.writer.close()

    def cos_knn(self, k, test_data, test_target, stored_data, stored_target):
        cosim = sklearn.metrics.pairwise.cosine_similarity(test_data, stored_data)
        top = [(heapq.nlargest((k), range(len(i)), i.take)) for i in cosim]
        top = [[stored_target[j] for j in i[:k]] for i in top]
        pred = [max(set(i), key=i.count) for i in top]
        pred = np.array(pred)
        precision = np.mean(pred == test_target)
        return precision

    def train(self, flags):
        if self.activate_load_model:
            model_path = os.path.join(flags.model_path, 'best_model_transfered.tar')
            if os.path.exists(model_path):
                print("Loading pretrained model at :%s to finetune" % model_path)
                self.load_state_dict(state_dict=model_path)
        time_start = datetime.datetime.now()

        # self.validate_workflow(self.batImageGenVals, flags, 0)

        for _ in range(flags.iteration_size):
            self.feature_extractor_network.train()
            if _ == 16000:
                self.opt_phi = torch.optim.Adam(self.phi_all.parameters(), lr=flags.lr / 100, amsgrad=True,
                                                weight_decay=self.weight_decay)
                self.opt = torch.optim.Adam(self.feature_extractor_network.parameters(), lr=flags.lr / 100,
                                            amsgrad=True,
                                            weight_decay=self.weight_decay)
            if _ == 8000:
                self.opt_phi = torch.optim.Adam(self.phi_all.parameters(), lr=flags.lr / 10, amsgrad=True,
                                                weight_decay=self.weight_decay)
                self.opt = torch.optim.Adam(self.feature_extractor_network.parameters(), lr=flags.lr / 10, amsgrad=True,
                                            weight_decay=self.weight_decay)
            total_loss = 0.0
            x_subset_list = []
            y_subset_list = []
            for i in range(self.num_train_domain):
                self.phi_all.train()
                images_train, labels_train = copy.deepcopy(self.batImageGenTrains[i].get_images_labels_batch())

                x_subset = torch.from_numpy(images_train.astype(np.float32))
                y_subset = torch.from_numpy(labels_train.astype(np.int64))
                y_subset += self.label_offset[i]
                # wrap the inputs and labels in Variable

                x_subset, y_subset = Variable(x_subset, requires_grad=False).cuda(), \
                                     Variable(y_subset, requires_grad=False).long().cuda()
                x_subset_list.append(x_subset)
                y_subset_list.append(y_subset)
                # y_pred = self.phi_all[i](self.feature_extractor_network(x_subset))
                # loss = self.ce_loss(y_pred, y_subset)
                # total_loss += loss
            x_set = torch.cat(x_subset_list, 0)
            y_set = torch.cat(y_subset_list, 0)
            mixed_x, y_set_a, y_set_b, lam = self.mixup_data(x_set, y_set, alpha=self.flags.mixup_alpha)
            y_pred = self.phi_all(self.feature_extractor_network(mixed_x))
            total_loss += self.mixup_criterion(y_pred, lam, y_set_a, y_set_b, method=self.flags.mixup_method)
            self.opt.zero_grad()
            self.opt_phi.zero_grad()
            total_loss.backward()
            self.opt.step()
            self.opt_phi.step()
            # print ('the iteration is %d, and loss in domain %s is %f.'%(_,self.domains_name[str(i)],loss.data.cpu().numpy()))
            if _ % 500 == 0 and flags.debug is True:
                time_end = datetime.datetime.now()
                epoch = (flags.iteration_size - int(_)) / 500
                time_cost = epoch * (time_end - time_start).seconds / 60
                time_start = time_end

                print('the number of iteration %d, and it is expected to take another %d minutes to complete..' % (
                    _, time_cost))
                self.validate_workflow(self.batImageGenVals, flags, _)
                self.heldout_test(flags, iter=_)

    def validate_workflow(self, batImageGenVals, flags, ite):
        accuracies = []
        for count, batImageGenVal in enumerate(batImageGenVals):
            accuracy_val = self.test(batImageGenTest=batImageGenVal, flags=flags, ite=ite,
                                     log_dir=flags.logs, log_prefix='val_index_{}'.format(count), count=count)
            accuracies.append(accuracy_val)
        mean_acc = np.mean(accuracies)
        if mean_acc > self.best_accuracy_val:
            self.best_accuracy_val = mean_acc

            f = open(os.path.join(flags.logs, 'Best_val_mixup %.2f iter:%d.txt' % (self.flags.mixup_alpha, ite)),
                     mode='a')
            f.write('ite:{}, best val accuracy:{}\n'.format(ite, self.best_accuracy_val))
            f.close()

            if not os.path.exists(flags.model_path):
                os.mkdir(flags.model_path)

            outfile = os.path.join(flags.model_path,
                                   'best_model_mixup %s %.2f iter:%d.tar' % (
                                       self.flags.mixup_method, self.flags.mixup_alpha, ite))
            torch.save((self.feature_extractor_network.state_dict(), self.phi_all.state_dict()), outfile)

    def test(self, flags, ite, log_prefix, log_dir='logs/', batImageGenTest=None, count=0):
        self.feature_extractor_network.eval()
        self.phi_all.eval()
        if batImageGenTest is None:
            batImageGenTest = BatchImageGenerator(flags=flags, file_path='', stage='test', metatest=False,
                                                  b_unfold_label=False)

        images_test = batImageGenTest.images
        labels_test = batImageGenTest.labels
        # labels_test += (self.label_offset[count] * labels_test)
        threshold = 1000
        if len(images_test) > threshold:
            n_slices_test = len(images_test) // threshold
            indices_test = []
            for per_slice in range(n_slices_test - 1):
                indices_test.append(len(images_test) * (per_slice + 1) // n_slices_test)
            test_image_splits = np.split(images_test, indices_or_sections=indices_test)

            # Verify the splits are correct
            test_image_splits_2_whole = np.concatenate(test_image_splits)
            assert np.all(images_test == test_image_splits_2_whole)
            # split the test data into splits and test them one by one
            test_image_preds = []
            for test_image_split in test_image_splits:
                # print(len(test_image_split))
                test_image_split = get_image(test_image_split)
                # print (test_image_split[0].shape)
                images_test_split = torch.from_numpy(np.array(test_image_split, dtype=np.float32))
                images_test_split = Variable(images_test_split, requires_grad=False).cuda()

                classifier_out = self.phi_all(
                    self.feature_extractor_network(images_test_split)).data.cpu().numpy()
                test_image_preds.append(classifier_out)
            # concatenate the test predictions first
            predictions = np.concatenate(test_image_preds)
        else:
            images_test = torch.from_numpy(np.array(images_test, dtype=np.float32))
            images_test = Variable(images_test, requires_grad=False).cuda()
            predictions = self.phi_all[count](self.feature_extractor_network(images_test)).data.cpu().numpy()

        accuracy = compute_accuracy(predictions=predictions, labels=labels_test, label_offset=self.label_offset[count])
        print('----------accuracy test of domain %s----------:' % (self.domains_name[str(count)]), accuracy)

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        log_path = os.path.join(log_dir, '{}.txt'.format(log_prefix))
        write_log(str('ite:{}, accuracy:{}'.format(ite, accuracy)), log_path=log_path)
        return accuracy


class ModelBaseline_VD:
    def __init__(self, flags):
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.configure(flags)
        self.setup_path(flags)
        self.init_network_parameter(flags)

        if not os.path.exists(flags.logs):
            os.mkdir(flags.logs)
        if not os.path.exists(flags.model_path):
            os.mkdir(flags.model_path)

    def __del__(self):
        print('release source')

    def configure(self, flags):
        self.flags_log = os.path.join(flags.logs, '%s.txt' % (flags.method))
        self.model_store = os.path.join(flags.model_path, '%s.pkl' % (flags.method))
        self.activate_load_model = True
        self.writer = SummaryWriter()

    def setup_path(self, flags):
        self.best_accuracy_val = -1
        if flags.dataset == 'VD':
            self.domains_name = get_domain_name()
            data_folder, train_data, val_data, test_data = get_data_folder()
        else:
            assert flags.dataset == 'VD', 'The current heterogeous DG code uses VD dataset'
        self.train_paths = []
        for data in train_data:
            path = os.path.join(data_folder, data)
            self.train_paths.append(path)

        self.val_paths = []
        for data in val_data:
            path = os.path.join(data_folder, data)
            self.val_paths.append(path)

        self.test_paths = []
        for data in test_data:
            path = os.path.join(data_folder, data)
            self.test_paths.append(path)

        unseen_index = 6
        self.unseen_data_path = []
        index = unseen_index
        for data in test_data[unseen_index:]:
            path = os.path.join(data_folder, data)
            self.unseen_data_path.append(self.train_paths[index])
            self.unseen_data_path.append(self.val_paths[index])
            self.train_paths.remove(self.train_paths[index])
            self.val_paths.remove(self.val_paths[index])

        if not os.path.exists(flags.logs):
            os.mkdir(flags.logs)
        flags_log = os.path.join(flags.logs, 'path_log.txt')
        write_log(str(self.train_paths), flags_log)
        write_log(str(self.val_paths), flags_log)
        write_log(str(self.unseen_data_path), flags_log)

        self.batImageGenTrains = []
        for train_path in self.train_paths:
            batImageGenTrain = BatchImageGenerator(flags=flags, file_path=train_path, stage='train',
                                                   metatest=False, b_unfold_label=False)
            self.batImageGenTrains.append(batImageGenTrain)

        self.batImageGenTrains_metatest = []
        for train_path in self.train_paths:
            batImageGenTrain_metatest = BatchImageGenerator(flags=flags, file_path=train_path, stage='train',
                                                            metatest=True, b_unfold_label=False)
            self.batImageGenTrains_metatest.append(batImageGenTrain_metatest)

        self.batImageGenVals = []
        for val_path in self.val_paths:
            batImageGenVal = BatchImageGenerator(flags=flags, file_path=val_path, stage='val',
                                                 metatest=False, b_unfold_label=True)
            self.batImageGenVals.append(batImageGenVal)

        self.batImageGenTests = []
        for test_path in self.test_paths:
            batImageGenTest = BatchImageGenerator(flags=flags, file_path=test_path, stage='test',
                                                  metatest=False, b_unfold_label=False)
            self.batImageGenTests.append(batImageGenTest)

    def init_network_parameter(self, flags):
        self.weight_decay = 1e-4  # 3e-4
        self.batch_size = flags.batch_size

        self.h = 512  # 1000
        self.hh = 100
        self.num_domain = 10
        self.num_test_domain = 4
        self.num_train_domain = self.num_domain - self.num_test_domain
        ######################################################
        self.feature_extractor_network = resnet18(pretrained=True)
        self.param_optim_theta = freeze_layer(self.feature_extractor_network)
        # theta means the network parameter of feature extractor, from d (the size of input) to h(the size of feature layer).
        self.opt = torch.optim.Adam(self.param_optim_theta, lr=flags.lr, amsgrad=True, weight_decay=self.weight_decay)
        # phi means the classifer network parameter, from h (the output feature layer of input data) to c (the number of classes).
        # Here, each domain has a classifier network.
        self.phi_all = []
        # CIFAR-100
        phi_CIFAR_100 = classifier(100)
        self.phi_all.append(phi_CIFAR_100)
        # Daimler Ped
        phi_Daimler = classifier(2)
        self.phi_all.append(phi_Daimler)
        # GTSRB
        phi_GTSRB = classifier(43)
        self.phi_all.append(phi_GTSRB)
        # Omniglot
        phi_Omniglot = classifier(1623)
        self.phi_all.append(phi_Omniglot)
        # SVHN
        phi_SVHN = classifier(10)
        self.phi_all.append(phi_SVHN)
        # ImageNet
        phi_ImageNet = classifier(1000)
        self.phi_all.append(phi_ImageNet)
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.opt_phi = []
        for i in range(self.num_train_domain):
            self.opt_phi.append(torch.optim.Adam(self.phi_all[i].parameters(), lr=flags.lr, amsgrad=True,
                                                 weight_decay=self.weight_decay))

    def load_state_dict(self, state_dict=''):
        tmp = torch.load(state_dict)
        pretrained_dict = tmp[0]
        # load the new state dict
        self.feature_extractor_network.load_state_dict(pretrained_dict)

        for i in range(self.num_train_domain):
            self.phi_all[i].load_state_dict(tmp[1][i])

    def heldout_test(self, flags):
        # load the best model on the validation data
        model_path = os.path.join(flags.model_path, 'best_model.tar')
        self.load_state_dict(state_dict=model_path)

        # Set the svm parameters by cross-validation
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        for i in range(self.num_test_domain):
            self.feature_extractor_network.eval()
            # test domains
            clf = GridSearchCV(svm.SVC(), tuned_parameters, scoring='precision_macro', n_jobs=5)

            batImageGenTest_train = BatchImageGenerator(flags=flags, file_path=self.unseen_data_path[2 * i],
                                                        stage='test', metatest=False, b_unfold_label=False)
            images_train = batImageGenTest_train.images
            labels_train = batImageGenTest_train.labels
            threshold = 100
            if len(images_train) > threshold:

                n_slices_test = len(images_train) / threshold
                indices_test = []
                for per_slice in range(n_slices_test - 1):
                    indices_test.append(len(images_train) * (per_slice + 1) / n_slices_test)
                train_image_splits = np.split(images_train, indices_or_sections=indices_test)

            # Verify the splits are correct
            train_image_splits_2_whole = np.concatenate(train_image_splits)
            assert np.all(images_train == train_image_splits_2_whole)

            # split the test data into splits and test them one by one
            train_feature_output = []
            for train_image_split in train_image_splits:
                # print(len(test_image_split))
                train_image_split = get_image(train_image_split)
                # print (test_image_split[0].shape)
                train_image_split = torch.from_numpy(np.array(train_image_split, dtype=np.float32))
                train_image_split = Variable(train_image_split, requires_grad=False).cuda()

                feature_out = self.feature_extractor_network(train_image_split).data.cpu().numpy()
                train_feature_output.append(feature_out)

            # concatenate the test predictions first
            train_feature_output = np.concatenate(train_feature_output)
            clf.fit(train_feature_output, labels_train)
            torch.cuda.empty_cache()
            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            write_log('Best parameters set found on development set:', self.flags_log)
            write_log(clf.best_params_, self.flags_log)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()
            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")

            batImageGenTest_test = BatchImageGenerator(flags=flags, file_path=self.unseen_data_path[2 * i + 1],
                                                       stage='test', metatest=False, b_unfold_label=False)

            images_test = batImageGenTest_test.images
            labels_test = batImageGenTest_test.labels
            threshold = 100
            if len(images_test) > threshold:

                n_slices_test = len(images_test) / threshold
                indices_test = []
                for per_slice in range(n_slices_test - 1):
                    indices_test.append(len(images_test) * (per_slice + 1) / n_slices_test)
                test_image_splits = np.split(images_test, indices_or_sections=indices_test)

            # split the test data into splits and test them one by one
            test_classifier_output = []
            for test_image_split in test_image_splits:
                # print(len(test_image_split))
                test_image_split = get_image(test_image_split)
                # print (test_image_split[0].shape)
                test_image_split = torch.from_numpy(np.array(test_image_split, dtype=np.float32))
                test_image_split = Variable(test_image_split, requires_grad=False).cuda()
                feature_out = self.feature_extractor_network(test_image_split)

                classifier_out = clf.predict(feature_out.data.cpu().numpy())
                test_classifier_output.append(classifier_out)
            test_classifier_output = np.concatenate(test_classifier_output)
            torch.cuda.empty_cache()
            accuracy = classification_report(labels_test, test_classifier_output)
            print(accuracy)
            precision = np.mean(test_classifier_output == labels_test)
            print(precision)
            # accuracy
            accuracy_info = 'the test domain %s.\n' % (self.domains_name[str(i + self.num_train_domain)])
            flags_log = os.path.join(flags.logs, 'heldout_test_log.txt')
            write_log(accuracy_info, flags_log)
            write_log(clf.best_params_, flags_log)
            # write_log(accuracy, flags_log)
            write_log(precision, flags_log)
        self.writer.close()

    def train(self, flags):
        if self.activate_load_model:
            model_path = os.path.join(flags.model_path, 'best_model.tar')
            if os.path.exists(model_path):
                self.load_state_dict(state_dict=model_path)
        time_start = datetime.datetime.now()
        for _ in range(flags.iteration_size):
            self.feature_extractor_network.train()
            if _ == 16000:
                for i in range(self.num_train_domain):
                    self.opt_phi[i] = torch.optim.Adam(self.phi_all[i].parameters(), lr=flags.lr / 100, amsgrad=True,
                                                       weight_decay=self.weight_decay)
                self.opt = torch.optim.Adam(self.feature_extractor_network.parameters(), lr=flags.lr / 100,
                                            amsgrad=True,
                                            weight_decay=self.weight_decay)
            if _ == 8000:
                for i in range(self.num_train_domain):
                    self.opt_phi[i] = torch.optim.Adam(self.phi_all[i].parameters(), lr=flags.lr / 10, amsgrad=True,
                                                       weight_decay=self.weight_decay)
                self.opt = torch.optim.Adam(self.feature_extractor_network.parameters(), lr=flags.lr / 10, amsgrad=True,
                                            weight_decay=self.weight_decay)
            total_loss = 0.0
            for i in range(self.num_train_domain):
                self.phi_all[i].train()
                images_train, labels_train = self.batImageGenTrains[i].get_images_labels_batch()

                x_subset = torch.from_numpy(images_train.astype(np.float32))
                y_subset = torch.from_numpy(labels_train.astype(np.int64))
                # wrap the inputs and labels in Variable
                x_subset, y_subset = Variable(x_subset, requires_grad=False).cuda(), \
                                     Variable(y_subset, requires_grad=False).long().cuda()

                y_pred = self.phi_all[i](self.feature_extractor_network(x_subset))
                # id_pred = model_id(x_subset)
                # loss = ce_loss(y_pred+id_pred, y_subset)
                loss = self.ce_loss(y_pred, y_subset)
                total_loss += loss
            self.opt.zero_grad()
            for k in range(self.num_train_domain):
                self.opt_phi[k].zero_grad()
            total_loss.backward()
            self.opt.step()
            for k in range(self.num_train_domain):
                self.opt_phi[k].step()
            # print ('the iteration is %d, and loss in domain %s is %f.'%(_,self.domains_name[str(i)],loss.data.cpu().numpy()))
            if _ % 500 == 0 and flags.debug is True:
                time_end = datetime.datetime.now()
                epoch = (flags.iteration_size - int(_)) / 500
                time_cost = epoch * (time_end - time_start).seconds / 60
                time_start = time_end

                print('the number of iteration %d, and it is expected to take another %d minutes to complete..' % (
                    _, time_cost))
                self.validate_workflow(self.batImageGenVals, flags, _)

    def validate_workflow(self, batImageGenVals, flags, ite):
        accuracies = []
        for count, batImageGenVal in enumerate(batImageGenVals):
            accuracy_val = self.test(batImageGenTest=batImageGenVal, flags=flags, ite=ite,
                                     log_dir=flags.logs, log_prefix='val_index_{}'.format(count), count=count)
            accuracies.append(accuracy_val)
        mean_acc = np.mean(accuracies)
        if mean_acc > self.best_accuracy_val:
            self.best_accuracy_val = mean_acc

            f = open(os.path.join(flags.logs, 'Best_val.txt'), mode='a')
            f.write('ite:{}, best val accuracy:{}\n'.format(ite, self.best_accuracy_val))
            f.close()

            if not os.path.exists(flags.model_path):
                os.mkdir(flags.model_path)

            outfile = os.path.join(flags.model_path, 'best_model.tar')
            state_phi = []
            for i in range(self.num_train_domain):
                state_phi.append(self.phi_all[i].state_dict())
            if flags.method == 'baseline':
                torch.save((self.feature_extractor_network.state_dict(), state_phi), outfile)
            if flags.method == 'Feature_Critic':
                torch.save((self.feature_extractor_network.state_dict(), state_phi, self.omega.state_dict()), outfile)

    def test(self, flags, ite, log_prefix, log_dir='logs/', batImageGenTest=None, count=0):
        self.feature_extractor_network.eval()
        self.phi_all[count].eval()
        if batImageGenTest is None:
            batImageGenTest = BatchImageGenerator(flags=flags, file_path='', stage='test', metatest=False,
                                                  b_unfold_label=False)

        images_test = batImageGenTest.images
        labels_test = batImageGenTest.labels
        threshold = 1000
        if len(images_test) > threshold:
            n_slices_test = len(images_test) // threshold
            indices_test = []
            for per_slice in range(n_slices_test - 1):
                indices_test.append(len(images_test) * (per_slice + 1) // n_slices_test)
            test_image_splits = np.split(images_test, indices_or_sections=indices_test)

            # Verify the splits are correct
            test_image_splits_2_whole = np.concatenate(test_image_splits)
            assert np.all(images_test == test_image_splits_2_whole)
            # split the test data into splits and test them one by one
            test_image_preds = []
            for test_image_split in test_image_splits:
                # print(len(test_image_split))
                test_image_split = get_image(test_image_split)
                # print (test_image_split[0].shape)
                images_test_split = torch.from_numpy(np.array(test_image_split, dtype=np.float32))
                images_test_split = Variable(images_test_split, requires_grad=False).cuda()

                classifier_out = self.phi_all[count](
                    self.feature_extractor_network(images_test_split)).data.cpu().numpy()
                test_image_preds.append(classifier_out)
            # concatenate the test predictions first
            predictions = np.concatenate(test_image_preds)
        else:
            images_test = torch.from_numpy(np.array(images_test, dtype=np.float32))
            images_test = Variable(images_test, requires_grad=False).cuda()
            predictions = self.phi_all[count](self.feature_extractor_network(images_test)).data.cpu().numpy()

        accuracy = compute_accuracy(predictions=predictions, labels=labels_test)
        print('----------accuracy test of domain %s----------:' % (self.domains_name[str(count)]), accuracy)

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        log_path = os.path.join(log_dir, '{}.txt'.format(log_prefix))
        write_log(str('ite:{}, accuracy:{}'.format(ite, accuracy)), log_path=log_path)
        return accuracy


