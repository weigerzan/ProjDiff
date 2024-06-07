def get_default_lr(deg, steps, sigma_0, dataset_name):
    if dataset_name=='imagenet':
        if sigma_0 > 0:
            if 'sr' in deg:
                if steps==20:
                    lr=0.9
                elif steps==100:
                    lr=0.5
                else:
                    print('No default step-size, using 1.0, recommending re-tuning')
                    lr=1.0
            elif 'inp' in deg:
                if steps==20:
                    lr=4.0
                elif steps==100:
                    lr=1.0
                else:
                    print('No default step-size, using 1.0, recommending re-tuning')
                    lr=1.0
            elif 'deblur' in deg:
                if steps==20:
                    lr=0.5
                elif steps==100:
                    lr=0.1
                else:
                    print('No default step-size, using 1.0, recommending re-tuning')
                    lr=1.0
            else:
                print('No default step-size, using 1.0, recommending re-tuning')
                lr=1.0
        else:
            if 'sr' in deg:
                if steps==20:
                    lr=1.5
                elif steps==100:
                    lr=1.7
                else:
                    print('No default step-size, using 1.0, recommending re-tuning')
                    lr=1.0
            elif 'inp' in deg:
                if steps==20:
                    lr=1.1
                elif steps==100:
                    lr=1.1
                else:
                    print('No default step-size, using 1.0, recommending re-tuning')
                    lr=1.0
            elif 'deblur' in deg:
                if steps==20:
                    lr=0.9
                elif steps==100:
                    lr=0.9
                else:
                    print('No default step-size, using 1.0, recommending re-tuning')
                    lr=1.0
            else:
                print('No default step-size, using 1.0, recommending re-tuning')
                lr=1.0
    elif dataset_name == 'celeba':
        if sigma_0 > 0:
            if 'sr' in deg:
                if steps==100:
                    lr=0.4
                else:
                    print('No default step-size, using 1.0, recommending re-tuning')
                    lr=1.0
            elif 'inp' in deg:
                if steps==100:
                    lr=1.1
                else:
                    print('No default step-size, using 1.0, recommending re-tuning')
                    lr=1.0
            elif 'deblur' in deg:
                if steps==100:
                    lr=0.1
                else:
                    print('No default step-size, using 1.0, recommending re-tuning')
                    lr=1.0
            else:
                print('No default step-size, using 1.0, recommending re-tuning')
                lr=1.0
        else:
            if 'sr' in deg:
                if steps==100:
                    lr=0.7
                else:
                    print('No default step-size, using 1.0, recommending re-tuning')
                    lr=1.0
            elif 'inp' in deg:
                if steps==100:
                    lr=1.1
                else:
                    print('No default step-size, using 1.0, recommending re-tuning')
                    lr=1.0
            elif 'deblur' in deg:
                if steps==100:
                    lr=0.8
                else:
                    print('No default step-size, using 1.0, recommending re-tuning')
                    lr=1.0
            else:
                print('No default step-size, using 1.0, recommending re-tuning')
                lr=1.0
    elif dataset_name == 'ffhq':
        if sigma_0 > 0:
            if 'phase' in deg:
                if steps==1000:
                    lr=1.9
                else:
                    print('No default step-size, using 1.0, recommending re-tuning')
                    lr=1.0
            elif 'hdr' in deg:
                if steps==100:
                    lr=1.0
                else:
                    print('No default step-size, using 1.0, recommending re-tuning')
                    lr=1.0
            else:
                print('No default step-size, using 1.0, recommending re-tuning')
                lr=1.0
        else:
            if 'phase' in deg:
                if steps==1000:
                    lr=1.5
                else:
                    print('No default step-size, using 1.0, recommending re-tuning')
                    lr=1.0
            elif 'hdr' in deg:
                # print(steps)
                if steps==100:
                    lr=2.0
                else:
                    print('No default step-size, using 1.0, recommending re-tuning')
                    lr=1.0
            else:
                print('No default step-size, using 1.0, recommending re-tuning')
                lr=1.0
    else:
        print('No default step-size, using 1.0, recommending re-tuning')
        lr=1.0
    return lr