import cv2
import time

#background
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

#area of interest
checkout_zone = [(1050, 400), (1100, 450)]  

def calculate_average_time(video_path):
    cap = cv2.VideoCapture(video_path)

    customer_entry_times = []  
    customer_exit_times = []  
    customer_id = 0            

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #movement detect
        fg_mask = bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        #disp
        cv2.imshow("Foreground Mask", fg_mask)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 100:  
                #bound box
                (x, y, w, h) = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)

                #check object
                if checkout_zone[0][0] < center[0] < checkout_zone[1][0] and checkout_zone[0][1] < center[1] < checkout_zone[1][1]:
                    
                    customer_entry_times.append(time.time())

                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        
        cv2.rectangle(frame, checkout_zone[0], checkout_zone[1], (0, 0, 255), 2)

        
        cv2.imshow("Checkout Zone", frame)

        # Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    #calculation
    if len(customer_entry_times) > 1:
        time_differences = [customer_entry_times[i] - customer_entry_times[i - 1] for i in range(1, len(customer_entry_times))]
        average_time = sum(time_differences) / len(time_differences)
        print(f"Average time taken by customers to checkout: {average_time:.2f} seconds")
    else:
        print("No sufficient customer data for average time calculation.")

if __name__ == "__main__":
    video_path = 'fringestorez.mp4'  
    calculate_average_time(video_path)
