# Job Description Keyword Parser
As it turns out, keywords in resumes are just as relevant as they may be in search engines. 

Applicant Tracking Systems will typically count keywords found in a resume as a way to rank them, or help HR sift through large stacks of applicants. Often these keywords will come from the job description, resumes from other applicants that are a good fit, and HR's own keyword entries. 

As a result, I decided to build a process to optimize my resume around ATSs. I guess I'm a ATSO now also...

To optimize the resume, I gathered ~30 job descriptions that were in line with what I was looking for. I could then gather a bulk analysis and find the most relevant keywords across a large array of data. This allowed me to create a base resume optimized around many of these terms. From there, I've build a seperate similar tool that looks at a single job description and compares it with your resume. 

This way you can effectively take your baseline resume and fine-tune it to a specific job listing. 

## A Word On Determining Relevance
I've decided to determine "relevance" from a batch of data with 3 different techniques, combining them to ID what keywords I wanted to optimize around. 
1. Semantic relevance [XLNet language model](https://github.com/zihangdai/xlnet)
2. Total count of keywords across the data
3. My own eye as to what I feel is most relevant of the previous two outputs.

I'd started by using Google's BERT (I am an SEO after all) but ended up moving to XLNet due to better performance / just overall more modern and improved technology. 

## Google Colab Docs
- [Bulk Job Description Parser](https://colab.research.google.com/drive/1zKRsSsZKFX4nfdyf4KtQQP_Z7lp3ixMY?usp=sharing)
- [Single Job Description Parser](https://colab.research.google.com/drive/1HSxDGj16Kn1hdpmwEef-dNGY9lscLQqx?usp=sharing)
