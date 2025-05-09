graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 593
  arrival_time 12241.529215656858
  lifetime 11.269303633135321
  num_nodes 14
  type "path"
  node [
    id 0
    label "0"
    cpu 7
    gpu 50
    rom 33
  ]
  node [
    id 1
    label "1"
    cpu 40
    gpu 5
    rom 19
  ]
  node [
    id 2
    label "2"
    cpu 19
    gpu 11
    rom 44
  ]
  node [
    id 3
    label "3"
    cpu 12
    gpu 32
    rom 49
  ]
  node [
    id 4
    label "4"
    cpu 18
    gpu 6
    rom 19
  ]
  node [
    id 5
    label "5"
    cpu 30
    gpu 1
    rom 21
  ]
  node [
    id 6
    label "6"
    cpu 6
    gpu 48
    rom 7
  ]
  node [
    id 7
    label "7"
    cpu 9
    gpu 34
    rom 12
  ]
  node [
    id 8
    label "8"
    cpu 10
    gpu 49
    rom 7
  ]
  node [
    id 9
    label "9"
    cpu 38
    gpu 14
    rom 40
  ]
  node [
    id 10
    label "10"
    cpu 49
    gpu 22
    rom 30
  ]
  node [
    id 11
    label "11"
    cpu 12
    gpu 28
    rom 12
  ]
  node [
    id 12
    label "12"
    cpu 27
    gpu 41
    rom 5
  ]
  node [
    id 13
    label "13"
    cpu 6
    gpu 32
    rom 26
  ]
  edge [
    source 0
    target 1
    bw 44
  ]
  edge [
    source 1
    target 2
    bw 12
  ]
  edge [
    source 2
    target 3
    bw 6
  ]
  edge [
    source 3
    target 4
    bw 4
  ]
  edge [
    source 4
    target 5
    bw 49
  ]
  edge [
    source 5
    target 6
    bw 22
  ]
  edge [
    source 6
    target 7
    bw 35
  ]
  edge [
    source 7
    target 8
    bw 38
  ]
  edge [
    source 8
    target 9
    bw 30
  ]
  edge [
    source 9
    target 10
    bw 33
  ]
  edge [
    source 10
    target 11
    bw 9
  ]
  edge [
    source 11
    target 12
    bw 25
  ]
  edge [
    source 12
    target 13
    bw 26
  ]
]
