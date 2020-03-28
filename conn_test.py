
def conv_humans_to_recs(humans, video_id, frame_id):
        import numpy as np
        import pandas as pd
        # Prepare the data for all humans to insert into DB
        #video_id, frame_num,human_num,keypoint_id, part_x, part_y, confidence
        #keypoint_id = human.body_parts.part_idx
        #part_x = human.body_parts.x
        #part_y = human.body_parts.y
        #confidence = human.body_parts.score
        human_num = []
        keypoint_id = []
        part_x = []
        part_y = []
        confidence = []
        h_c = 0
        print ("CONN: Found ", len(humans), " humans")
        for human in humans:
            h_c +=1
            #print("CONN:trying for ", h_c, " human")
            try:
                for part_idx in range(18):
                    human_num.append(h_c)
                    #print ("\t Appended human num")
                    keypoint_id.append(part_idx)
                    #print ("\t Appended human num")
                    part_x.append(human.body_parts[part_idx].x)
                    part_y.append(human.body_parts[part_idx].y)
                    confidence.append(human.body_parts[part_idx].score)
            except:
                    pass

        num_keypoints = len(keypoint_id)
        video_id = [video_id for i in range(num_keypoints)]
        frame_id = [frame_id for i in range(num_keypoints)]
        #print("CONN_TEST:Found:", num_keypoints, " keypoints")
        #recs = np.core.records.fromarrays([video_id, frame_id, human_num, keypoint_id,part_x, part_y, confidence])
        df = pd.DataFrame({'video_id':video_id, 'frame_id':frame_id,'human_num':human_num,'keypoint_id':keypoint_id,'part_x':part_x, 'part_y':part_y, 'confidence':confidence})
        recs = [list(row) for row in df.itertuples(index=False)] 
        return recs

def insert_pose_keypoints_humans(humans, video_id, frame_id):
    import psycopg2
    import json
    import pandas as pd
    try:
        #read the JSON config file
        with open('db_creds.json') as json_file:
            creds= json.load(json_file)
            user = creds['user']
            pwd = creds['password']
            host = creds['host']
            port = creds['port']
            database = creds['database']
        
        connection = psycopg2.connect(user = user,
                                  password = pwd,
                                  host = host,
                                  port = port,
                                  database = database)
        #print ("CONN: Established Connection")
        cursor = connection.cursor()
        
        # Get the data in the required format
        data_recs = conv_humans_to_recs(humans, video_id, frame_id)
        #print ("CONN: Converted to recs")
        insert_query = 'INSERT INTO pose_data (video_id, frame_num, human_num, keypoint_id, part_x, part_y, confidence) VALUES (%s,%s,%s,%s,%s,%s,%s)'
        for rec in data_recs:
                cursor.execute(insert_query,rec)
        connection.commit()

    except (Exception, psycopg2.Error) as error :
        print ("Error in PostgreSQL try block:", error)
    finally:
    #closing database connection.
        if(connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
